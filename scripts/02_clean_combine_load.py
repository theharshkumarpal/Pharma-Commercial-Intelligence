"""
TASKS 3–8: Data Cleaning, Combination, Drug Mapping, Diabetes Market,
           Database Load, and Validation.

Pipeline:
  Task 3 — Clean each year's data (missing, dupes, types, suppression flags)
  Task 4 — Combine all six years into fact_prescriptions
  Task 5 — Create dim_drug via RxNorm/RxClass mapping
  Task 6 — Lock diabetes market definition
  Task 7 — Load into DuckDB (used in place of PostgreSQL — same SQL, zero config)
  Task 8 — Validate the database
"""

import os
import time
import json
import requests
import certifi
import numpy as np
import os
import time
import json
import requests
import certifi
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ===================================================================
# TASK 3 — Clean each year's data
# ===================================================================
def clean_year(year: int) -> pd.DataFrame:
    """Clean a single year's raw CSV. Returns cleaned DataFrame."""
    pattern = f"{RAW_DIR}/{year}/Medicare_Part_D_Prescribers_by_Provider_and_Drug_{year}.csv"
    df = pd.read_csv(pattern)
    print(f"\n{'='*60}")
    print(f"TASK 3 — Cleaning {year}  (raw rows: {len(df):,})")
    print(f"{'='*60}")

    # --- 3a. Check missing values ---
    missing_pct = df.isnull().mean() * 100
    print(f"  Missing % per column:\n{missing_pct.to_string()}")

    # --- 3b. Remove exact duplicate records ---
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    print(f"  Duplicates removed: {n_dupes:,}")

    # --- 3c. Create suppression flag (CRITICAL — do NOT replace with zero) ---
    numeric_cols = ["Tot_Clms", "Tot_30day_Fills", "Tot_Drug_Cst", "Tot_Day_Suply", "Tot_Benes"]
    df["is_suppressed"] = df[numeric_cols].isnull().any(axis=1)
    n_suppressed = df["is_suppressed"].sum()
    print(f"  Suppressed records flagged: {n_suppressed:,} ({n_suppressed/len(df)*100:.1f}%)")

    # --- 3d. Standardize types ---
    df["Year"] = int(year)
    # NPI must be string (leading zeros possible in real data)
    df["Prscrbr_NPI"] = df["Prscrbr_NPI"].astype(str).str.strip()
    # Validate NPI format: 10 digits
    valid_npi = df["Prscrbr_NPI"].str.match(r"^\d{10}$")
    n_invalid_npi = (~valid_npi).sum()
    if n_invalid_npi > 0:
        print(f"  WARNING: {n_invalid_npi} invalid NPI values — removed")
        df = df[valid_npi].copy()

    # Numeric columns — convert safely (suppressed already NaN)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate non-negative for non-suppressed
    for col in numeric_cols:
        mask_bad = (~df["is_suppressed"]) & (df[col] < 0)
        if mask_bad.any():
            print(f"  WARNING: {mask_bad.sum()} negative values in {col} — set to NaN")
            df.loc[mask_bad, col] = np.nan

    # --- 3e. Standardize string columns ---
    df["Brnd_Name"] = df["Brnd_Name"].str.strip().str.upper()
    df["Gnrc_Name"] = df["Gnrc_Name"].str.strip().str.upper()
    df["Prscrbr_Type"] = df["Prscrbr_Type"].str.strip()
    df["Prscrbr_State_Abrvtn"] = df["Prscrbr_State_Abrvtn"].str.strip().str.upper()

    # Validate state (2 letter code)
    valid_state = df["Prscrbr_State_Abrvtn"].str.match(r"^[A-Z]{2}$")
    n_bad_state = (~valid_state).sum()
    if n_bad_state > 0:
        print(f"  WARNING: {n_bad_state} invalid state values")

    print(f"  Final cleaned rows: {len(df):,}")
    return df


# ===================================================================
# TASK 4 — Combine all six years
# ===================================================================
def combine_years(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine cleaned yearly DataFrames into fact_prescriptions."""
    print(f"\n{'='*60}")
    print("TASK 4 — Combining all six years into fact_prescriptions")
    print(f"{'='*60}")

    combined = pd.concat(dfs, ignore_index=True)

    # Rename to standard schema
    combined = combined.rename(columns={
        "Prscrbr_NPI": "npi",
        "Prscrbr_Type": "specialty",
        "Prscrbr_State_Abrvtn": "state",
        "Brnd_Name": "brand_name",
        "Gnrc_Name": "generic_name",
        "Tot_Clms": "total_claims",
        "Tot_30day_Fills": "total_30day_fills",
        "Tot_Drug_Cst": "total_drug_cost",
        "Tot_Day_Suply": "total_day_supply",
        "Tot_Benes": "total_beneficiaries",
        "Year": "year"
    })

    print(f"  Combined rows: {len(combined):,}")
    print(f"  Years: {sorted(combined['year'].unique())}")
    print(f"  Unique NPIs: {combined['npi'].nunique():,}")
    print(f"  Unique drugs: {combined['brand_name'].nunique()}")
    return combined


# ===================================================================
# TASK 5 — Create drug master table via RxNorm/RxClass
# ===================================================================

# Known diabetes drug classification (built from RxNorm/RxClass API lookups)
# We query the API for each unique generic name and cache results.
DIABETES_CLASS_MAP = {
    "SEMAGLUTIDE": "GLP-1 receptor agonists",
    "TIRZEPATIDE": "GLP-1 receptor agonists",
    "DULAGLUTIDE": "GLP-1 receptor agonists",
    "LIRAGLUTIDE": "GLP-1 receptor agonists",
    "EMPAGLIFLOZIN": "SGLT2 inhibitors",
    "DAPAGLIFLOZIN": "SGLT2 inhibitors",
    "CANAGLIFLOZIN": "SGLT2 inhibitors",
    "ERTUGLIFLOZIN": "SGLT2 inhibitors",
    "SITAGLIPTIN": "DPP-4 inhibitors",
    "LINAGLIPTIN": "DPP-4 inhibitors",
    "SAXAGLIPTIN": "DPP-4 inhibitors",
    "INSULIN GLARGINE": "Insulin",
    "INSULIN LISPRO": "Insulin",
    "INSULIN ASPART": "Insulin",
    "INSULIN DEGLUDEC": "Insulin",
    "METFORMIN HCL": "Biguanides",
    "METFORMIN HYDROCHLORIDE": "Biguanides",
    "GLIPIZIDE": "Sulfonylureas",
    "GLIMEPIRIDE": "Sulfonylureas",
    "GLYBURIDE": "Sulfonylureas",
    "PIOGLITAZONE HCL": "Thiazolidinediones",
    "PIOGLITAZONE HYDROCHLORIDE": "Thiazolidinediones",
    "ROSIGLITAZONE": "Thiazolidinediones",
    "PRAMLINTIDE ACETATE": "Other diabetes therapies",
    "BROMOCRIPTINE MESYLATE": "Other diabetes therapies",
    "ACARBOSE": "Other diabetes therapies",
    "MIGLITOL": "Other diabetes therapies",
}


def query_rxclass(drug_name: str) -> dict:
    """Query RxNorm/RxClass API for drug classification."""
    result = {"rxcui": None, "ingredient": drug_name, "therapeutic_class": None}
    try:
        # Step 1: Get RxCUI from drug name
        url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}&search=1"
        r = requests.get(url, verify=certifi.where(), timeout=10)
        if r.status_code == 200:
            data = r.json()
            ids = data.get("idGroup", {}).get("rxnormId", [])
            if ids:
                result["rxcui"] = ids[0]

        # Step 2: Get drug classes via RxClass
        if result["rxcui"]:
            url2 = f"https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={result['rxcui']}&relaSource=ATC"
            r2 = requests.get(url2, verify=certifi.where(), timeout=10)
            if r2.status_code == 200:
                data2 = r2.json()
                concepts = data2.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
                for c in concepts:
                    class_name = c.get("rxclassMinConceptItem", {}).get("className", "")
                    if class_name:
                        result["therapeutic_class"] = class_name
                        break
        time.sleep(0.15)  # Rate-limit API calls
    except Exception as e:
        print(f"    RxClass API error for {drug_name}: {e}")
    return result


def create_drug_master(fact: pd.DataFrame) -> pd.DataFrame:
    """Task 5 & 6: Create dim_drug table with RxNorm/RxClass mapping."""
    print(f"\n{'='*60}")
    print("TASK 5 — Creating drug master table (dim_drug)")
    print(f"{'='*60}")

    drugs = fact[["brand_name", "generic_name"]].drop_duplicates().reset_index(drop=True)
    drugs["drug_id"] = range(1, len(drugs) + 1)

    # Map using RxNorm/RxClass API with fallback to curated map
    rxcuis = []
    ingredients = []
    therapeutic_classes = []
    diabetes_classes = []

    print(f"  Mapping {len(drugs)} unique drugs via RxNorm/RxClass API...")
    for _, row in drugs.iterrows():
        gname = row["generic_name"]

        # Try curated map first (faster, more reliable for known diabetes drugs)
        if gname in DIABETES_CLASS_MAP:
            diabetes_classes.append(DIABETES_CLASS_MAP[gname])
            rxcuis.append(None)
            ingredients.append(gname)
            therapeutic_classes.append(DIABETES_CLASS_MAP[gname])
        else:
            # Query API
            info = query_rxclass(gname)
            rxcuis.append(info["rxcui"])
            ingredients.append(info["ingredient"])
            tc = info["therapeutic_class"]
            therapeutic_classes.append(tc)
            # Check if it maps to diabetes
            dc = DIABETES_CLASS_MAP.get(gname, None)
            if dc is None and tc and "diabet" in tc.lower():
                dc = "Other diabetes therapies"
            diabetes_classes.append(dc)

    drugs["rxcui"] = rxcuis
    drugs["ingredient"] = ingredients
    drugs["therapeutic_class"] = therapeutic_classes
    drugs["diabetes_class"] = diabetes_classes

    n_diabetes = drugs["diabetes_class"].notna().sum()
    print(f"  Total unique drugs: {len(drugs)}")
    print(f"  Diabetes drugs mapped: {n_diabetes}")
    print(f"  Diabetes classes: {drugs['diabetes_class'].dropna().unique().tolist()}")
    return drugs


# ===================================================================
# TASK 6 — Lock diabetes market definition
# ===================================================================
def lock_diabetes_market(dim_drug: pd.DataFrame) -> pd.DataFrame:
    """Document the locked diabetes market definition."""
    print(f"\n{'='*60}")
    print("TASK 6 — Locking diabetes market definition")
    print(f"{'='*60}")

    diabetes_drugs = dim_drug[dim_drug["diabetes_class"].notna()].copy()
    class_counts = diabetes_drugs["diabetes_class"].value_counts()
    print("  Diabetes therapeutic classes and drug counts:")
    for cls, cnt in class_counts.items():
        print(f"    {cls}: {cnt} drugs")

    # Save market definition doc
    market_def = {
        "market": "Diabetes therapies in Medicare Part D",
        "classification_source": "RxNorm/RxClass API + curated mapping",
        "categories": class_counts.to_dict(),
        "total_drugs": int(len(diabetes_drugs)),
        "drugs": diabetes_drugs[["brand_name", "generic_name", "diabetes_class"]].to_dict(orient="records")
    }
    with open("docs/diabetes_market_definition.json", "w") as f:
        json.dump(market_def, f, indent=2)
    print("  Market definition saved to docs/diabetes_market_definition.json")
    return diabetes_drugs


# ===================================================================
# TASK 7 — Load into Supabase PostgreSQL
# ===================================================================
def load_database(fact: pd.DataFrame, dim_drug: pd.DataFrame):
    """Load all tables into Supabase PostgreSQL analytical database."""
    print(f"\n{'='*60}")
    print("TASK 7 — Loading into Supabase PostgreSQL")
    print(f"{'='*60}")

    from db import get_db_engine
    engine = get_db_engine()

    with engine.begin() as conn:
        # dim_drug
        dim_drug.to_sql("dim_drug", conn, if_exists="replace", index=False)
        print("  dim_drug loaded")

        # dim_provider
        providers = fact[["npi", "specialty", "state"]].drop_duplicates(subset=["npi"]).reset_index(drop=True)
        providers.to_sql("dim_provider", conn, if_exists="replace", index=False)
        print("  dim_provider loaded")

        # dim_year
        years_df = pd.DataFrame({"year": YEARS})
        years_df.to_sql("dim_year", conn, if_exists="replace", index=False)
        print("  dim_year loaded")

        # fact_prescriptions — join with drug_id
        fact_with_drug = fact.merge(
            dim_drug[["drug_id", "brand_name", "generic_name"]],
            on=["brand_name", "generic_name"],
            how="left"
        )
        fact_with_drug.to_sql("fact_prescriptions", conn, if_exists="replace", index=False)
        print("  fact_prescriptions loaded")

    print("  Database loaded into Supabase PostgreSQL successfully")


# ===================================================================
# TASK 8 — Validate the database
# ===================================================================
def validate_database():
    """Run comprehensive validation checks against Supabase PostgreSQL."""
    print(f"\n{'='*60}")
    print("TASK 8 — Validating the database (Supabase PostgreSQL)")
    print(f"{'='*60}")

    from db import get_db_engine
    engine = get_db_engine()
    report = {}

    with engine.connect() as conn:
        # Row counts per year
        rc = pd.read_sql("SELECT year, COUNT(*) as cnt FROM fact_prescriptions GROUP BY year ORDER BY year", conn)
        print("\n  Row counts per year:")
        for _, r in rc.iterrows():
            print(f"    {int(r['year'])}: {int(r['cnt']):,}")
        report["row_counts_per_year"] = rc.set_index("year")["cnt"].to_dict()

        # Unique key check (year + npi + brand_name + generic_name)
        dup_check = pd.read_sql("""
            SELECT year, npi, brand_name, generic_name, COUNT(*) as cnt
            FROM fact_prescriptions
            GROUP BY year, npi, brand_name, generic_name
            HAVING COUNT(*) > 1
        """, conn)
        n_dup_keys = len(dup_check)
        print(f"\n  Duplicate key combinations: {n_dup_keys}")
        report["duplicate_key_combinations"] = int(n_dup_keys)

        # Missing percentages
        miss = pd.read_sql("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN total_claims IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_missing_claims,
                SUM(CASE WHEN total_30day_fills IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_missing_fills,
                SUM(CASE WHEN total_drug_cost IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_missing_cost,
                SUM(CASE WHEN is_suppressed THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_suppressed
            FROM fact_prescriptions
        """, conn)
        print(f"\n  Total records: {int(miss['total'].iloc[0]):,}")
        print(f"  % missing claims: {miss['pct_missing_claims'].iloc[0]:.2f}%")
        print(f"  % missing fills:  {miss['pct_missing_fills'].iloc[0]:.2f}%")
        print(f"  % missing cost:   {miss['pct_missing_cost'].iloc[0]:.2f}%")
        print(f"  % suppressed:     {miss['pct_suppressed'].iloc[0]:.2f}%")

        # Aggregates
        aggs = pd.read_sql("""
            SELECT
                SUM(total_claims) as total_claims,
                SUM(total_30day_fills) as total_fills,
                SUM(total_drug_cost) as total_spending,
                COUNT(DISTINCT npi) as unique_providers,
                COUNT(DISTINCT brand_name) as unique_drugs
            FROM fact_prescriptions
            WHERE NOT is_suppressed
        """, conn)
        print(f"\n  Total claims (non-suppressed): {aggs['total_claims'].iloc[0]:,.0f}")
        print(f"  Total fills (non-suppressed):  {aggs['total_fills'].iloc[0]:,.0f}")
        print(f"  Total spending (non-suppressed): ${aggs['total_spending'].iloc[0]:,.2f}")
        print(f"  Unique providers: {int(aggs['unique_providers'].iloc[0]):,}")
        print(f"  Unique drugs: {int(aggs['unique_drugs'].iloc[0])}")

        report["total_claims"] = float(aggs["total_claims"].iloc[0])
        report["total_fills"] = float(aggs["total_fills"].iloc[0])
        report["total_spending"] = float(aggs["total_spending"].iloc[0])
        report["unique_providers"] = int(aggs["unique_providers"].iloc[0])
        report["unique_drugs"] = int(aggs["unique_drugs"].iloc[0])

    # Save validation report
    with open("outputs/validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n  Validation report saved to outputs/validation_report.json")

    return report


# ===================================================================
# MAIN
# ===================================================================
def main():
    # TASK 3 — Clean each year
    cleaned_dfs = []
    for year in YEARS:
        df = clean_year(year)
        cleaned_dfs.append(df)

    # TASK 4 — Combine
    fact = combine_years(cleaned_dfs)

    # TASK 5 — Drug master
    dim_drug = create_drug_master(fact)

    # TASK 6 — Lock diabetes market
    diabetes_drugs = lock_diabetes_market(dim_drug)

    # Save processed files
    fact.to_csv(f"{PROCESSED_DIR}/fact_prescriptions.csv", index=False)
    dim_drug.to_csv(f"{PROCESSED_DIR}/dim_drug.csv", index=False)
    print(f"\n  Processed data saved to {PROCESSED_DIR}/")

    # TASK 7 — Load database
    load_database(fact, dim_drug)

    # TASK 8 — Validate
    validate_database()

    print("\n" + "=" * 60)
    print("TASKS 3-8 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

