"""
TASKS 23–28: Drug Adoption, Geographic Opportunity, Opportunity Score
  Task 23 — Define HCP-level therapy adoption intensity
  Task 24 — Identify adoption trends
  Task 25 — State-level analytics
  Task 26 — Growth/share opportunity matrix
  Task 27 — Analyst-defined opportunity score
  Task 28 — Sensitivity analysis
"""

import json
import numpy as np
import pandas as pd
import duckdb

DB_PATH = "data/hcproject.duckdb"


# ===================================================================
# TASK 23 — Define HCP-level therapy adoption
# ===================================================================
def task23_adoption_definition(con) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 23 — Defining HCP-level therapy adoption intensity")
    print(f"{'='*60}")

    adoption = con.execute("""
        SELECT
            f.npi,
            f.year,
            f.specialty,
            f.state,
            d.diabetes_class,
            SUM(f.total_30day_fills) as therapy_fills
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class IS NOT NULL
          AND f.is_suppressed = false
        GROUP BY f.npi, f.year, f.specialty, f.state, d.diabetes_class
    """).fetchdf()

    # Total diabetes fills per HCP per year
    hcp_totals = adoption.groupby(["npi", "year"])["therapy_fills"].sum().reset_index()
    hcp_totals = hcp_totals.rename(columns={"therapy_fills": "total_diabetes_fills"})

    adoption = adoption.merge(hcp_totals, on=["npi", "year"])
    adoption["adoption_share"] = (adoption["therapy_fills"] / adoption["total_diabetes_fills"] * 100).round(2)

    print(f"  Adoption records: {len(adoption):,}")
    print(f"  Metric: HCP-level therapy adoption intensity")
    print(f"  Formula: therapy_fills / total_diabetes_fills × 100")
    print(f"  NOTE: This is NOT individual patient adoption — it is provider-level prescribing intensity.")
    return adoption


# ===================================================================
# TASK 24 — Adoption trends
# ===================================================================
def task24_adoption_trends(adoption: pd.DataFrame, hcp: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("TASK 24 — Identifying adoption trends")
    print(f"{'='*60}")

    trends = {}

    # By Year × Therapy
    yt = adoption.groupby(["year", "diabetes_class"]).agg(
        mean_adoption=("adoption_share", "mean"),
        total_fills=("therapy_fills", "sum"),
        n_hcps=("npi", "nunique")
    ).reset_index()
    print("\n  Year × Therapy adoption trends (mean adoption share %):")
    pivot_yt = yt.pivot_table(index="diabetes_class", columns="year",
                              values="mean_adoption", aggfunc="mean").round(1)
    print(pivot_yt.to_string())
    trends["year_therapy"] = yt

    # By Specialty (2024)
    spec = adoption[adoption["year"] == 2024].groupby(["specialty", "diabetes_class"]).agg(
        mean_adoption=("adoption_share", "mean"),
        n_hcps=("npi", "nunique")
    ).reset_index()
    print("\n  Top specialty adoption rates for GLP-1 (2024):")
    glp1_spec = spec[spec["diabetes_class"] == "GLP-1 receptor agonists"].nlargest(5, "mean_adoption")
    print(glp1_spec[["specialty", "mean_adoption", "n_hcps"]].to_string(index=False))
    trends["specialty"] = spec

    # By State (2024)
    st = adoption[adoption["year"] == 2024].groupby(["state", "diabetes_class"]).agg(
        mean_adoption=("adoption_share", "mean"),
        n_hcps=("npi", "nunique")
    ).reset_index()
    trends["state"] = st

    # By HCP segment (if available)
    if "segment_name" in hcp.columns:
        adoption_2024 = adoption[adoption["year"] == 2024].copy()
        adoption_2024["npi"] = adoption_2024["npi"].astype(str)
        hcp_temp = hcp[["npi", "segment_name"]].drop_duplicates().copy()
        hcp_temp["npi"] = hcp_temp["npi"].astype(str)
        adoption_seg = adoption_2024.merge(
            hcp_temp,
            on="npi", how="left"
        )
        seg = adoption_seg.groupby(["segment_name", "diabetes_class"]).agg(
            mean_adoption=("adoption_share", "mean"),
            n_hcps=("npi", "nunique")
        ).reset_index()
        print("\n  Adoption by HCP segment (GLP-1, 2024):")
        glp1_seg = seg[seg["diabetes_class"] == "GLP-1 receptor agonists"]
        print(glp1_seg[["segment_name", "mean_adoption", "n_hcps"]].to_string(index=False))
        trends["segment"] = seg

    return trends


# ===================================================================
# TASK 25 — State-level analytics
# ===================================================================
def task25_state_analytics(con) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 25 — Building state-level analytics")
    print(f"{'='*60}")

    state_analytics = con.execute("""
        SELECT
            f.state,
            f.year,
            SUM(f.total_30day_fills) as total_fills,
            SUM(f.total_drug_cost) as total_spending,
            COUNT(DISTINCT f.npi) as num_hcps,
            AVG(f.total_30day_fills) as avg_hcp_volume
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class IS NOT NULL
          AND f.is_suppressed = false
        GROUP BY f.state, f.year
        ORDER BY f.state, f.year
    """).fetchdf()

    # Add GLP-1 specific fills
    glp1_state = con.execute("""
        SELECT
            f.state,
            f.year,
            SUM(f.total_30day_fills) as glp1_fills
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class = 'GLP-1 receptor agonists'
          AND f.is_suppressed = false
        GROUP BY f.state, f.year
    """).fetchdf()

    state_analytics = state_analytics.merge(glp1_state, on=["state", "year"], how="left")
    state_analytics["glp1_fills"] = state_analytics["glp1_fills"].fillna(0)
    state_analytics["glp1_market_share"] = (
        state_analytics["glp1_fills"] / state_analytics["total_fills"] * 100
    ).round(2)

    # Market share and growth
    totals_by_year = state_analytics.groupby("year")["total_fills"].sum().reset_index()
    totals_by_year = totals_by_year.rename(columns={"total_fills": "national_fills"})
    state_analytics = state_analytics.merge(totals_by_year, on="year")
    state_analytics["state_market_share"] = (
        state_analytics["total_fills"] / state_analytics["national_fills"] * 100
    ).round(2)

    # YoY growth per state
    state_analytics = state_analytics.sort_values(["state", "year"])
    state_analytics["yoy_growth"] = state_analytics.groupby("state")["total_fills"].pct_change() * 100
    state_analytics["yoy_growth"] = state_analytics["yoy_growth"].round(2)

    print(f"  States with data: {state_analytics['state'].nunique()}")
    s2024 = state_analytics[state_analytics["year"] == 2024].nlargest(10, "total_fills")
    print("\n  Top 10 states by diabetes fills (2024):")
    print(s2024[["state", "total_fills", "state_market_share", "glp1_market_share",
                  "yoy_growth", "num_hcps"]].to_string(index=False))
    return state_analytics


# ===================================================================
# TASK 26 — Growth/share opportunity matrix
# ===================================================================
def task26_growth_share_matrix(state_analytics: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 26 — Growth/share opportunity matrix (GLP-1 focus)")
    print(f"{'='*60}")

    s2024 = state_analytics[state_analytics["year"] == 2024].copy()

    # Classify states into quadrants
    med_growth = s2024["yoy_growth"].median()
    med_share = s2024["glp1_market_share"].median()

    def classify(row):
        high_growth = row["yoy_growth"] > med_growth
        high_share = row["glp1_market_share"] > med_share
        if high_growth and high_share:
            return "Priority"
        elif high_growth and not high_share:
            return "Emerging"
        elif not high_growth and high_share:
            return "Mature"
        else:
            return "Low priority"

    s2024["quadrant"] = s2024.apply(classify, axis=1)

    print(f"  Median growth threshold: {med_growth:.1f}%")
    print(f"  Median GLP-1 market share threshold: {med_share:.1f}%")
    print("\n  Quadrant distribution:")
    for q, cnt in s2024["quadrant"].value_counts().items():
        states_in_q = s2024[s2024["quadrant"] == q]["state"].tolist()
        print(f"    {q}: {cnt} states — {states_in_q[:5]}")

    return s2024


# ===================================================================
# TASK 27 — Opportunity score
# ===================================================================
def task27_opportunity_score(hcp: pd.DataFrame, state_analytics: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 27 — Creating analyst-defined opportunity score")
    print(f"{'='*60}")

    opp = hcp.copy()

    # Get 2024 state growth
    s2024 = state_analytics[state_analytics["year"] == 2024][["state", "yoy_growth", "glp1_market_share"]]
    s2024 = s2024.rename(columns={"yoy_growth": "state_growth", "glp1_market_share": "state_glp1_share"})
    opp = opp.merge(s2024, on="state", how="left")

    def normalize_0_100(series):
        """Normalize series to 0-100 range."""
        s = series.fillna(0)
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(50, index=series.index)
        return ((s - mn) / (mx - mn) * 100).round(2)

    # Normalize each component
    opp["volume_score"] = normalize_0_100(opp["total_fills"])
    opp["growth_score"] = normalize_0_100(opp["yoy_growth"])

    # Therapy affinity = GLP-1 + SGLT2 share (newer therapies)
    if "glp1_share" in opp.columns and "sglt2_share" in opp.columns:
        opp["therapy_affinity_raw"] = opp["glp1_share"] + opp["sglt2_share"]
    else:
        opp["therapy_affinity_raw"] = 0
    opp["therapy_affinity_score"] = normalize_0_100(opp["therapy_affinity_raw"])

    opp["geo_growth_score"] = normalize_0_100(opp["state_growth"])
    opp["market_opportunity_score"] = normalize_0_100(opp["state_glp1_share"])

    # Weighted composite
    opp["opportunity_score"] = (
        0.30 * opp["volume_score"] +
        0.25 * opp["growth_score"] +
        0.20 * opp["therapy_affinity_score"] +
        0.15 * opp["geo_growth_score"] +
        0.10 * opp["market_opportunity_score"]
    ).round(2)

    print(f"  Weights: Volume=30%, Growth=25%, Therapy Affinity=20%, "
          f"Geo Growth=15%, Market Opportunity=10%")
    print(f"\n  Top 10 HCPs by Opportunity Score:")
    top10 = opp.nlargest(10, "opportunity_score")
    print(top10[["npi", "specialty", "state", "total_fills", "yoy_growth",
                  "opportunity_score"]].to_string(index=False))

    return opp


# ===================================================================
# TASK 28 — Sensitivity analysis
# ===================================================================
def task28_sensitivity(opp: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("TASK 28 — Sensitivity analysis")
    print(f"{'='*60}")

    base_top100 = set(opp.nlargest(100, "opportunity_score")["npi"])

    scenarios = {
        "Base (30/25/20/15/10)": [0.30, 0.25, 0.20, 0.15, 0.10],
        "Volume-heavy (45/15/15/15/10)": [0.45, 0.15, 0.15, 0.15, 0.10],
        "Growth-heavy (15/45/15/15/10)": [0.15, 0.45, 0.15, 0.15, 0.10],
        "Balanced (20/20/20/20/20)": [0.20, 0.20, 0.20, 0.20, 0.20],
        "Therapy-focused (20/20/35/15/10)": [0.20, 0.20, 0.35, 0.15, 0.10],
    }

    components = ["volume_score", "growth_score", "therapy_affinity_score",
                   "geo_growth_score", "market_opportunity_score"]

    results = {}
    for name, weights in scenarios.items():
        score = sum(w * opp[c] for w, c in zip(weights, components))
        top100 = set(opp.assign(alt_score=score).nlargest(100, "alt_score")["npi"])
        overlap = len(base_top100 & top100)
        stability = overlap / 100 * 100
        results[name] = {
            "weights": dict(zip(components, weights)),
            "top100_overlap_with_base": overlap,
            "stability_pct": stability
        }
        print(f"  {name}: {overlap}/100 overlap ({stability:.0f}% stable)")

    # Save
    with open("outputs/sensitivity_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Sensitivity analysis saved to outputs/sensitivity_analysis.json")
    return results


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    hcp = pd.read_csv("outputs/hcp_features.csv")

    adoption = task23_adoption_definition(con)
    trends = task24_adoption_trends(adoption, hcp)
    state_analytics = task25_state_analytics(con)
    growth_share = task26_growth_share_matrix(state_analytics)
    opp = task27_opportunity_score(hcp, state_analytics)
    sensitivity = task28_sensitivity(opp)

    con.close()

    # Save outputs
    state_analytics.to_csv("outputs/state_opportunity.csv", index=False)
    growth_share.to_csv("outputs/growth_share_matrix.csv", index=False)
    opp.to_csv("outputs/hcp_opportunity.csv", index=False)

    print(f"\n{'='*60}")
    print("TASKS 23-28 COMPLETE — Adoption, geography, and opportunity saved")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
