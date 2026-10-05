"""
TASKS 14–22: HCP Analytics & Segmentation
  Task 14 — HCP-level features
  Task 15 — Focal-therapy features
  Task 16 — HCP growth
  Task 17 — HCP behavioral dataset
  Task 18 — Prepare ML features
  Task 19 — Handle outliers
  Task 20 — Determine optimal K
  Task 21 — Run K-Means
  Task 22 — Interpret clusters
"""

import json
import warnings
import numpy as np
import pandas as pd
import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

DB_PATH = "data/hcproject.duckdb"


# ===================================================================
# TASK 14 — HCP-level features
# ===================================================================
def task14_hcp_features(con) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 14 — Creating HCP-level features")
    print(f"{'='*60}")

    hcp = con.execute("""
        SELECT
            f.npi,
            f.state,
            f.specialty,
            SUM(f.total_30day_fills) as total_fills,
            SUM(f.total_claims) as total_claims,
            SUM(f.total_drug_cost) as total_cost,
            COUNT(DISTINCT f.brand_name) as unique_drugs,
            COUNT(DISTINCT d.diabetes_class) as unique_classes
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class IS NOT NULL
          AND f.is_suppressed = false
        GROUP BY f.npi, f.state, f.specialty
    """).fetchdf()

    # Therapy share = each HCP's diabetes fills / total diabetes fills
    total_market_fills = hcp["total_fills"].sum()
    hcp["therapy_share"] = (hcp["total_fills"] / total_market_fills * 100).round(4)

    print(f"  HCPs with diabetes prescribing: {len(hcp):,}")
    print(f"  Mean fills per HCP: {hcp['total_fills'].mean():,.1f}")
    print(f"  Median fills per HCP: {hcp['total_fills'].median():,.1f}")
    return hcp


# ===================================================================
# TASK 15 — Focal-therapy features
# ===================================================================
def task15_focal_therapy(con, hcp: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 15 — Focal-therapy features")
    print(f"{'='*60}")

    focal = con.execute("""
        SELECT
            f.npi,
            d.diabetes_class,
            SUM(f.total_30day_fills) as class_fills
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class IS NOT NULL
          AND f.is_suppressed = false
        GROUP BY f.npi, d.diabetes_class
    """).fetchdf()

    # Pivot to get fills per class per HCP
    pivot = focal.pivot_table(index="npi", columns="diabetes_class",
                              values="class_fills", aggfunc="sum", fill_value=0)

    # Standardize column names
    class_cols = {
        "GLP-1 receptor agonists": "glp1_fills",
        "SGLT2 inhibitors": "sglt2_fills",
        "Insulin": "insulin_fills",
        "DPP-4 inhibitors": "dpp4_fills",
        "Biguanides": "biguanide_fills",
        "Sulfonylureas": "sulfonylurea_fills",
        "Thiazolidinediones": "tzd_fills",
        "Other diabetes therapies": "other_fills",
    }

    for orig_name, new_name in class_cols.items():
        if orig_name in pivot.columns:
            pivot = pivot.rename(columns={orig_name: new_name})
        else:
            pivot[new_name] = 0

    pivot = pivot.reset_index()

    # Merge with HCP features
    hcp = hcp.merge(pivot, on="npi", how="left")

    # Calculate shares
    hcp_total = hcp["total_fills"]
    for fill_col, share_col in [
        ("glp1_fills", "glp1_share"),
        ("sglt2_fills", "sglt2_share"),
        ("insulin_fills", "insulin_share"),
    ]:
        if fill_col in hcp.columns:
            hcp[share_col] = np.where(hcp_total > 0, (hcp[fill_col] / hcp_total * 100).round(2), 0)

    print(f"  GLP-1 share mean: {hcp['glp1_share'].mean():.1f}%")
    print(f"  SGLT2 share mean: {hcp['sglt2_share'].mean():.1f}%")
    print(f"  Insulin share mean: {hcp['insulin_share'].mean():.1f}%")
    return hcp


# ===================================================================
# TASK 16 — HCP growth
# ===================================================================
def task16_hcp_growth(con, hcp: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 16 — HCP growth")
    print(f"{'='*60}")

    # Get fills by year per HCP
    yearly = con.execute("""
        SELECT f.npi, f.year, SUM(f.total_30day_fills) as year_fills
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class IS NOT NULL
          AND f.is_suppressed = false
        GROUP BY f.npi, f.year
    """).fetchdf()

    # 2023 vs 2024 growth
    y23 = yearly[yearly["year"] == 2023].set_index("npi")["year_fills"].rename("fills_2023")
    y24 = yearly[yearly["year"] == 2024].set_index("npi")["year_fills"].rename("fills_2024")
    growth = pd.concat([y23, y24], axis=1).reset_index()

    # Only calculate growth where baseline > 0
    growth["yoy_growth"] = np.where(
        growth["fills_2023"] > 0,
        ((growth["fills_2024"] - growth["fills_2023"]) / growth["fills_2023"] * 100).round(2),
        np.nan
    )

    # Longer-term growth (2019 vs 2024)
    y19 = yearly[yearly["year"] == 2019].set_index("npi")["year_fills"].rename("fills_2019")
    lt = pd.concat([y19, y24], axis=1).reset_index()
    lt["lt_growth"] = np.where(
        lt["fills_2019"] > 0,
        ((lt["fills_2024"] - lt["fills_2019"]) / lt["fills_2019"] * 100).round(2),
        np.nan
    )

    hcp = hcp.merge(growth[["npi", "yoy_growth"]], on="npi", how="left")
    hcp = hcp.merge(lt[["npi", "lt_growth"]], on="npi", how="left")

    print(f"  HCPs with YoY growth data: {hcp['yoy_growth'].notna().sum():,}")
    print(f"  Mean YoY growth: {hcp['yoy_growth'].mean():.2f}%")
    print(f"  Median YoY growth: {hcp['yoy_growth'].median():.2f}%")
    return hcp


# ===================================================================
# TASK 17 — HCP behavioral dataset
# ===================================================================
def task17_behavioral_dataset(hcp: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 17 — HCP behavioral dataset")
    print(f"{'='*60}")

    # Brand share: non-generic drugs / total (estimate: branded = drugs with distinct brand vs generic)
    # We'll approximate brand share as the share of fills from branded drugs
    # (In our data, drugs like OZEMPIC, JARDIANCE etc. are branded; METFORMIN, GLIPIZIDE are generic)
    # Use fills from high-cost drugs as proxy for brand share
    branded_cols = ["glp1_fills", "sglt2_fills", "dpp4_fills", "insulin_fills"]
    avail_cols = [c for c in branded_cols if c in hcp.columns]
    hcp["brand_fills"] = hcp[avail_cols].sum(axis=1)
    hcp["brand_share"] = np.where(
        hcp["total_fills"] > 0,
        (hcp["brand_fills"] / hcp["total_fills"] * 100).round(2),
        0
    )

    # Final ML table
    ml_cols = ["npi", "specialty", "state", "total_fills", "yoy_growth",
               "therapy_share", "unique_drugs", "brand_share",
               "glp1_share", "sglt2_share", "insulin_share"]
    available = [c for c in ml_cols if c in hcp.columns]
    hcp_behavioral = hcp[available].copy()

    print(f"  Behavioral dataset shape: {hcp_behavioral.shape}")
    print(f"  Columns: {list(hcp_behavioral.columns)}")
    print(hcp_behavioral.describe().round(2).to_string())
    return hcp


# ===================================================================
# TASKS 18-22 — HCP Segmentation (K-Means)
# ===================================================================
def task18_22_segmentation(hcp: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 18 — Preparing ML features")
    print(f"{'='*60}")

    # TASK 18 — Select exactly these core features
    feature_cols = ["total_fills", "yoy_growth", "therapy_share", "unique_drugs", "brand_share"]
    ml_data = hcp[["npi"] + feature_cols].copy()

    # Drop rows with NaN in feature columns
    ml_data = ml_data.dropna(subset=feature_cols)
    print(f"  Records for clustering: {len(ml_data):,}")

    # TASK 19 — Handle outliers: log-transform fills (highly skewed)
    print(f"\n{'='*60}")
    print("TASK 19 — Handling outliers")
    print(f"{'='*60}")
    ml_data["total_fills_log"] = np.log1p(ml_data["total_fills"])
    print(f"  Fills skewness before log: {ml_data['total_fills'].skew():.2f}")
    print(f"  Fills skewness after log:  {ml_data['total_fills_log'].skew():.2f}")

    # Use log-transformed fills for clustering
    cluster_features = ["total_fills_log", "yoy_growth", "therapy_share", "unique_drugs", "brand_share"]
    X = ml_data[cluster_features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Features standardized: {cluster_features}")

    # TASK 20 — Determine optimal K
    print(f"\n{'='*60}")
    print("TASK 20 — Determining optimal K")
    print(f"{'='*60}")

    k_range = [2, 3, 4, 5, 6]
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        print(f"  K={k}: Inertia={km.inertia_:,.0f}, Silhouette={sil:.4f}")

    best_k_idx = np.argmax(silhouettes)
    # Balance interpretability: prefer K=4 if silhouette is close to max
    best_k = k_range[best_k_idx]
    if best_k == 2 and silhouettes[2] > 0.85 * silhouettes[0]:
        best_k = 4  # 4 is more interpretable for pharma segmentation
    print(f"\n  Selected K={best_k} (best interpretable solution)")

    # TASK 21 — Run K-Means
    print(f"\n{'='*60}")
    print(f"TASK 21 — Running K-Means (K={best_k})")
    print(f"{'='*60}")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    ml_data["cluster"] = km_final.fit_predict(X_scaled)
    print(f"  Cluster sizes:")
    for c, cnt in ml_data["cluster"].value_counts().sort_index().items():
        print(f"    Cluster {c}: {cnt:,} HCPs")

    # TASK 22 — Interpret clusters
    print(f"\n{'='*60}")
    print("TASK 22 — Interpreting clusters")
    print(f"{'='*60}")

    cluster_profiles = ml_data.groupby("cluster")[feature_cols].agg(["mean", "median"]).round(2)
    print("\n  Cluster feature means:")

    cluster_means = ml_data.groupby("cluster")[feature_cols].mean().round(2)
    print(cluster_means.to_string())

    # Auto-assign business names based on characteristics
    cluster_names = {}
    for c in range(best_k):
        row = cluster_means.loc[c]
        if row["total_fills"] > cluster_means["total_fills"].quantile(0.75) and row["yoy_growth"] > 0:
            name = "High-Volume Core Prescribers"
        elif row["yoy_growth"] > cluster_means["yoy_growth"].quantile(0.75) and row["brand_share"] > cluster_means["brand_share"].median():
            name = "Emerging Therapy Adopters"
        elif row["unique_drugs"] > cluster_means["unique_drugs"].quantile(0.6):
            name = "Diversified Prescribers"
        elif row["total_fills"] < cluster_means["total_fills"].quantile(0.3):
            name = "Low-Volume / Low-Growth Prescribers"
        else:
            name = f"Moderate-Volume Prescribers (Segment {c})"
        cluster_names[c] = name
        print(f"\n  Cluster {c} → '{name}'")
        print(f"    Fills: {row['total_fills']:,.0f}, Growth: {row['yoy_growth']:.1f}%, "
              f"Drugs: {row['unique_drugs']:.1f}, Brand: {row['brand_share']:.1f}%")

    ml_data["segment_name"] = ml_data["cluster"].map(cluster_names)

    # Merge cluster assignments back to main HCP table
    hcp = hcp.merge(ml_data[["npi", "cluster", "segment_name"]], on="npi", how="left")

    # Save cluster analysis
    cluster_analysis = {
        "k_selected": best_k,
        "silhouette_scores": dict(zip([str(k) for k in k_range], silhouettes)),
        "inertias": dict(zip([str(k) for k in k_range], inertias)),
        "cluster_names": cluster_names,
        "cluster_means": cluster_means.to_dict(),
    }
    with open("outputs/cluster_analysis.json", "w") as f:
        json.dump(cluster_analysis, f, indent=2, default=str)

    return hcp


def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    hcp = task14_hcp_features(con)
    hcp = task15_focal_therapy(con, hcp)
    hcp = task16_hcp_growth(con, hcp)
    hcp = task17_behavioral_dataset(hcp)
    hcp = task18_22_segmentation(hcp)

    con.close()

    # Save HCP features & segments
    hcp.to_csv("outputs/hcp_features.csv", index=False)
    hcp.to_parquet("outputs/hcp_features.parquet", index=False)

    print(f"\n{'='*60}")
    print("TASKS 14-22 COMPLETE — HCP analytics & segmentation saved")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
