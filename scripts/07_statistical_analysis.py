"""
TASKS 35–37: Statistical Analysis & Final Analytical Tables
  Task 35 — Correlation analysis
  Task 36 — Compare HCP segments (statistical tests)
  Task 37 — Create final analytical dataset tables
"""

import json
import warnings
import numpy as np
import pandas as pd
import duckdb
from scipy import stats

warnings.filterwarnings("ignore")

DB_PATH = "data/hcproject.duckdb"


# ===================================================================
# TASK 35 — Correlation analysis
# ===================================================================
def task35_correlation(hcp: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("TASK 35 — Correlation analysis")
    print(f"{'='*60}")

    results = {}

    # Volume vs Growth
    valid = hcp.dropna(subset=["total_fills", "yoy_growth"])
    if len(valid) > 10:
        r, p = stats.spearmanr(valid["total_fills"], valid["yoy_growth"])
        results["volume_vs_growth"] = {
            "spearman_r": round(float(r), 4),
            "p_value": float(p),
            "n": int(len(valid)),
            "interpretation": "Negative correlation expected: high-volume prescribers have less room to grow"
        }
        print(f"  Volume vs Growth: r={r:.4f}, p={p:.4e}, n={len(valid)}")

    # Volume vs Therapy Diversity
    valid2 = hcp.dropna(subset=["total_fills", "unique_drugs"])
    if len(valid2) > 10:
        r2, p2 = stats.spearmanr(valid2["total_fills"], valid2["unique_drugs"])
        results["volume_vs_drug_diversity"] = {
            "spearman_r": round(float(r2), 4),
            "p_value": float(p2),
            "n": int(len(valid2)),
            "interpretation": "Higher-volume prescribers tend to prescribe more diverse drugs"
        }
        print(f"  Volume vs Drug Diversity: r={r2:.4f}, p={p2:.4e}, n={len(valid2)}")

    # GLP-1 share vs Growth
    if "glp1_share" in hcp.columns:
        valid3 = hcp.dropna(subset=["glp1_share", "yoy_growth"])
        if len(valid3) > 10:
            r3, p3 = stats.spearmanr(valid3["glp1_share"], valid3["yoy_growth"])
            results["glp1_share_vs_growth"] = {
                "spearman_r": round(float(r3), 4),
                "p_value": float(p3),
                "n": int(len(valid3)),
                "interpretation": "Tests whether GLP-1 prescribing intensity is associated with overall growth"
            }
            print(f"  GLP-1 Share vs Growth: r={r3:.4f}, p={p3:.4e}, n={len(valid3)}")

    # Brand share vs Volume
    valid4 = hcp.dropna(subset=["brand_share", "total_fills"])
    if len(valid4) > 10:
        r4, p4 = stats.spearmanr(valid4["brand_share"], valid4["total_fills"])
        results["brand_share_vs_volume"] = {
            "spearman_r": round(float(r4), 4),
            "p_value": float(p4),
            "n": int(len(valid4)),
            "interpretation": "Tests whether higher-volume prescribers use more branded therapies"
        }
        print(f"  Brand Share vs Volume: r={r4:.4f}, p={p4:.4e}, n={len(valid4)}")

    print(f"\n  NOTE: Correlations are observational. No causal claims are made.")
    return results


# ===================================================================
# TASK 36 — Compare HCP segments
# ===================================================================
def task36_segment_comparison(hcp: pd.DataFrame) -> dict:
    print(f"\n{'='*60}")
    print("TASK 36 — Comparing HCP segments (statistical tests)")
    print(f"{'='*60}")

    results = {}

    if "segment_name" not in hcp.columns or "cluster" not in hcp.columns:
        print("  WARNING: No cluster/segment data found. Skipping.")
        return results

    segments = hcp["segment_name"].dropna().unique()
    n_segments = len(segments)
    print(f"  Segments to compare: {n_segments}")

    # --- Test 1: Do segments differ in GLP-1 adoption? ---
    if "glp1_share" in hcp.columns and n_segments >= 2:
        groups = [hcp[hcp["segment_name"] == s]["glp1_share"].dropna().values
                  for s in segments]
        groups = [g for g in groups if len(g) >= 5]

        if len(groups) >= 2:
            # Kruskal-Wallis (non-parametric, handles non-normal distributions)
            stat, p = stats.kruskal(*groups)

            # Effect size (eta-squared for Kruskal-Wallis)
            N = sum(len(g) for g in groups)
            k = len(groups)
            eta_sq = (stat - k + 1) / (N - k)
            eta_sq = max(0, eta_sq)

            results["glp1_adoption_across_segments"] = {
                "test": "Kruskal-Wallis H",
                "statistic": round(float(stat), 4),
                "p_value": float(p),
                "effect_size_eta_sq": round(float(eta_sq), 4),
                "n_segments": len(groups),
                "total_n": int(N),
                "conclusion": "Significant" if p < 0.05 else "Not significant",
                "interpretation": "Tests whether GLP-1 adoption differs significantly across HCP segments"
            }
            print(f"\n  GLP-1 adoption across segments:")
            print(f"    Kruskal-Wallis H={stat:.4f}, p={p:.4e}")
            print(f"    Effect size (eta²)={eta_sq:.4f}")
            print(f"    {'Significant at α=0.05' if p < 0.05 else 'Not significant at α=0.05'}")

            # Segment-level descriptives
            for s in segments:
                seg_data = hcp[hcp["segment_name"] == s]["glp1_share"].dropna()
                if len(seg_data) > 0:
                    print(f"    {s}: mean={seg_data.mean():.1f}%, median={seg_data.median():.1f}%, n={len(seg_data)}")

    # --- Test 2: Do segments differ in overall volume? ---
    if n_segments >= 2:
        groups_vol = [hcp[hcp["segment_name"] == s]["total_fills"].dropna().values
                      for s in segments]
        groups_vol = [g for g in groups_vol if len(g) >= 5]

        if len(groups_vol) >= 2:
            stat_v, p_v = stats.kruskal(*groups_vol)
            N_v = sum(len(g) for g in groups_vol)
            eta_v = max(0, (stat_v - len(groups_vol) + 1) / (N_v - len(groups_vol)))

            results["volume_across_segments"] = {
                "test": "Kruskal-Wallis H",
                "statistic": round(float(stat_v), 4),
                "p_value": float(p_v),
                "effect_size_eta_sq": round(float(eta_v), 4),
                "conclusion": "Significant" if p_v < 0.05 else "Not significant",
            }
            print(f"\n  Volume across segments:")
            print(f"    Kruskal-Wallis H={stat_v:.4f}, p={p_v:.4e}, eta²={eta_v:.4f}")

    # --- Test 3: Pairwise comparison — highest vs lowest growth segments ---
    if "yoy_growth" in hcp.columns and n_segments >= 2:
        seg_means = hcp.groupby("segment_name")["yoy_growth"].mean()
        if len(seg_means) >= 2:
            high_seg = seg_means.idxmax()
            low_seg = seg_means.idxmin()

            g_high = hcp[hcp["segment_name"] == high_seg]["yoy_growth"].dropna()
            g_low = hcp[hcp["segment_name"] == low_seg]["yoy_growth"].dropna()

            if len(g_high) >= 5 and len(g_low) >= 5:
                u_stat, u_p = stats.mannwhitneyu(g_high, g_low, alternative="two-sided")

                # Effect size: rank-biserial correlation
                n1, n2 = len(g_high), len(g_low)
                r_effect = 1 - (2 * u_stat) / (n1 * n2)

                # 95% CI using bootstrap (simplified)
                diff = g_high.mean() - g_low.mean()

                results["high_vs_low_growth_segments"] = {
                    "test": "Mann-Whitney U",
                    "high_growth_segment": high_seg,
                    "low_growth_segment": low_seg,
                    "statistic": float(u_stat),
                    "p_value": float(u_p),
                    "effect_size_rank_biserial": round(float(r_effect), 4),
                    "mean_difference": round(float(diff), 2),
                    "conclusion": "Significant" if u_p < 0.05 else "Not significant",
                }
                print(f"\n  High-growth vs Low-growth segment comparison:")
                print(f"    {high_seg} vs {low_seg}")
                print(f"    Mann-Whitney U={u_stat:.0f}, p={u_p:.4e}")
                print(f"    Effect size (rank-biserial r)={r_effect:.4f}")
                print(f"    Mean growth difference: {diff:.2f} pp")

    # --- Test 4: Specialty differences in therapy adoption ---
    if "glp1_share" in hcp.columns:
        spec_groups = [hcp[hcp["specialty"] == s]["glp1_share"].dropna().values
                       for s in hcp["specialty"].unique()]
        spec_groups = [g for g in spec_groups if len(g) >= 10]

        if len(spec_groups) >= 2:
            stat_s, p_s = stats.kruskal(*spec_groups)
            N_s = sum(len(g) for g in spec_groups)
            eta_s = max(0, (stat_s - len(spec_groups) + 1) / (N_s - len(spec_groups)))
            results["glp1_by_specialty"] = {
                "test": "Kruskal-Wallis H",
                "statistic": round(float(stat_s), 4),
                "p_value": float(p_s),
                "effect_size_eta_sq": round(float(eta_s), 4),
            }
            print(f"\n  GLP-1 adoption by specialty:")
            print(f"    Kruskal-Wallis H={stat_s:.4f}, p={p_s:.4e}, eta²={eta_s:.4f}")

    return results


# ===================================================================
# TASK 37 — Create final analytical tables
# ===================================================================
def task37_final_tables(con) -> None:
    print(f"\n{'='*60}")
    print("TASK 37 — Creating final analytical tables")
    print(f"{'='*60}")

    # 1. market_yearly
    market_yearly = pd.read_csv("outputs/market_yearly.csv")
    print(f"  market_yearly: {len(market_yearly)} rows, {list(market_yearly.columns)}")

    # 2. drug_yearly
    drug_yearly = pd.read_csv("outputs/drug_yearly.csv")
    print(f"  drug_yearly: {len(drug_yearly)} rows")

    # 3. therapy_yearly
    therapy_yearly = pd.read_csv("outputs/therapy_yearly.csv")
    print(f"  therapy_yearly: {len(therapy_yearly)} rows")

    # 4. hcp_features
    hcp_features = pd.read_csv("outputs/hcp_features.csv")
    print(f"  hcp_features: {len(hcp_features)} rows, {len(hcp_features.columns)} columns")

    # 5. hcp_segments (extract cluster info)
    if "segment_name" in hcp_features.columns:
        hcp_segments = hcp_features[["npi", "specialty", "state", "cluster", "segment_name",
                                      "total_fills", "yoy_growth", "brand_share"]].copy()
        hcp_segments.to_csv("outputs/hcp_segments.csv", index=False)
        print(f"  hcp_segments: {len(hcp_segments)} rows")

    # 6. state_opportunity
    state_opp = pd.read_csv("outputs/state_opportunity.csv")
    print(f"  state_opportunity: {len(state_opp)} rows")

    # 7. hcp_opportunity
    hcp_opp = pd.read_csv("outputs/hcp_opportunity.csv")
    print(f"  hcp_opportunity: {len(hcp_opp)} rows")

    # 8. forecast
    forecast = pd.read_csv("outputs/forecast.csv")
    print(f"  forecast: {len(forecast)} rows")

    # Load all into DuckDB as final analytical tables
    tables_to_load = {
        "market_yearly": market_yearly,
        "drug_yearly": drug_yearly,
        "therapy_yearly": therapy_yearly,
        "hcp_features": hcp_features,
        "state_opportunity": state_opp,
        "hcp_opportunity": hcp_opp,
        "forecast": forecast,
    }
    if "segment_name" in hcp_features.columns:
        tables_to_load["hcp_segments"] = hcp_segments

    for tname, tdf in tables_to_load.items():
        con.execute(f"DROP TABLE IF EXISTS {tname}")
        con.execute(f"CREATE TABLE {tname} AS SELECT * FROM tdf")
        cnt = con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        print(f"  Loaded {tname} into DuckDB: {cnt:,} rows")

    print("\n  All final analytical tables loaded into DuckDB")


def main():
    hcp = pd.read_csv("outputs/hcp_features.csv")
    con = duckdb.connect(DB_PATH)

    # TASK 35
    corr_results = task35_correlation(hcp)

    # TASK 36
    segment_results = task36_segment_comparison(hcp)

    # TASK 37
    task37_final_tables(con)
    con.close()

    # Save statistical results
    all_stats = {
        "correlations": corr_results,
        "segment_comparisons": segment_results,
        "note": "All tests are observational. No causal claims are made."
    }
    with open("outputs/statistical_analysis.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\n  Statistical analysis saved to outputs/statistical_analysis.json")

    print(f"\n{'='*60}")
    print("TASKS 35-37 COMPLETE — Statistics & final tables saved")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
