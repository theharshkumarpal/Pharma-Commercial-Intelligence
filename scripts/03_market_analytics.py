"""
TASKS 9–13: Market Analytics
  Task 9  — Annual market size
  Task 10 — Market growth (YoY)
  Task 11 — Drug market share
  Task 12 — Therapeutic-class performance
  Task 13 — Growth leaders
"""

import json
import numpy as np
import pandas as pd
import duckdb

DB_PATH = "data/hcproject.duckdb"


def get_diabetes_fact(con) -> pd.DataFrame:
    """Get fact_prescriptions filtered to diabetes drugs only, non-suppressed."""
    return con.execute("""
        SELECT f.*
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class IS NOT NULL
          AND f.is_suppressed = false
    """).fetchdf()


# ===================================================================
# TASK 9 — Annual market size
# ===================================================================
def task9_market_size(con) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 9 — Annual market size")
    print(f"{'='*60}")

    market_yearly = con.execute("""
        SELECT
            f.year,
            SUM(f.total_claims) as total_claims,
            SUM(f.total_30day_fills) as total_fills,
            SUM(f.total_drug_cost) as total_spending,
            COUNT(DISTINCT f.npi) as num_prescribers,
            COUNT(DISTINCT f.brand_name) as num_drugs
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class IS NOT NULL
          AND f.is_suppressed = false
        GROUP BY f.year
        ORDER BY f.year
    """).fetchdf()

    print(market_yearly.to_string(index=False))
    return market_yearly


# ===================================================================
# TASK 10 — Market growth (YoY)
# ===================================================================
def task10_market_growth(market_yearly: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 10 — Market growth (YoY)")
    print(f"{'='*60}")

    my = market_yearly.sort_values("year").copy()
    for col in ["total_claims", "total_fills", "total_spending"]:
        prev = my[col].shift(1)
        my[f"{col}_yoy_growth"] = ((my[col] - prev) / prev * 100).round(2)

    print(my[["year", "total_claims_yoy_growth", "total_fills_yoy_growth",
              "total_spending_yoy_growth"]].to_string(index=False))
    return my


# ===================================================================
# TASK 11 — Drug market share
# ===================================================================
def task11_drug_market_share(con, market_yearly: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 11 — Drug market share")
    print(f"{'='*60}")

    drug_yearly = con.execute("""
        SELECT
            f.year,
            f.brand_name,
            f.generic_name,
            d.diabetes_class,
            SUM(f.total_30day_fills) as drug_fills,
            SUM(f.total_drug_cost) as drug_spending,
            SUM(f.total_claims) as drug_claims
        FROM fact_prescriptions f
        JOIN dim_drug d ON f.drug_id = d.drug_id
        WHERE d.diabetes_class IS NOT NULL
          AND f.is_suppressed = false
        GROUP BY f.year, f.brand_name, f.generic_name, d.diabetes_class
        ORDER BY f.year, drug_fills DESC
    """).fetchdf()

    # Merge total fills for market share calculation
    drug_yearly = drug_yearly.merge(
        market_yearly[["year", "total_fills", "total_spending"]],
        on="year", how="left"
    )
    drug_yearly["fills_market_share"] = (drug_yearly["drug_fills"] / drug_yearly["total_fills"] * 100).round(3)
    drug_yearly["spending_market_share"] = (drug_yearly["drug_spending"] / drug_yearly["total_spending"] * 100).round(3)

    # Top 10 by fills in 2024
    top2024 = drug_yearly[drug_yearly["year"] == 2024].nlargest(10, "drug_fills")
    print("  Top 10 drugs by fills (2024):")
    print(top2024[["brand_name", "drug_fills", "fills_market_share", "spending_market_share"]].to_string(index=False))
    return drug_yearly


# ===================================================================
# TASK 12 — Therapeutic-class performance
# ===================================================================
def task12_class_performance(drug_yearly: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 12 — Therapeutic-class performance")
    print(f"{'='*60}")

    therapy_yearly = drug_yearly.groupby(["year", "diabetes_class"]).agg(
        class_fills=("drug_fills", "sum"),
        class_spending=("drug_spending", "sum"),
        class_claims=("drug_claims", "sum"),
    ).reset_index()

    # Market share per year
    yearly_totals = therapy_yearly.groupby("year")["class_fills"].transform("sum")
    therapy_yearly["market_share"] = (therapy_yearly["class_fills"] / yearly_totals * 100).round(2)

    # YoY growth per class
    therapy_yearly = therapy_yearly.sort_values(["diabetes_class", "year"])
    therapy_yearly["yoy_growth"] = therapy_yearly.groupby("diabetes_class")["class_fills"].pct_change() * 100
    therapy_yearly["yoy_growth"] = therapy_yearly["yoy_growth"].round(2)

    # Average cost per fill
    therapy_yearly["avg_cost_per_fill"] = (therapy_yearly["class_spending"] / therapy_yearly["class_fills"]).round(2)

    print(therapy_yearly[therapy_yearly["year"] == 2024][
        ["diabetes_class", "class_fills", "class_spending", "market_share", "yoy_growth", "avg_cost_per_fill"]
    ].to_string(index=False))

    return therapy_yearly


# ===================================================================
# TASK 13 — Growth leaders
# ===================================================================
def task13_growth_leaders(drug_yearly: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("TASK 13 — Growth leaders")
    print(f"{'='*60}")

    # Calculate YoY growth per drug
    dy = drug_yearly.sort_values(["brand_name", "year"]).copy()
    dy["prev_fills"] = dy.groupby("brand_name")["drug_fills"].shift(1)
    dy["abs_growth"] = dy["drug_fills"] - dy["prev_fills"]
    dy["pct_growth"] = np.where(
        dy["prev_fills"] > 0,
        (dy["abs_growth"] / dy["prev_fills"] * 100),
        np.nan
    )

    # For meaningful scale: filter drugs with ≥ 1000 fills
    recent = dy[(dy["year"] == 2024) & (dy["drug_fills"] >= 500)].copy()

    print("  === Largest market share (2024) ===")
    top_share = recent.nlargest(5, "fills_market_share")
    print(top_share[["brand_name", "drug_fills", "fills_market_share"]].to_string(index=False))

    print("\n  === Fastest % growth (2024, with scale) ===")
    fast_growers = recent.dropna(subset=["pct_growth"]).nlargest(5, "pct_growth")
    print(fast_growers[["brand_name", "drug_fills", "pct_growth", "abs_growth"]].to_string(index=False))

    print("\n  === Largest absolute growth (2024) ===")
    abs_growers = recent.dropna(subset=["abs_growth"]).nlargest(5, "abs_growth")
    print(abs_growers[["brand_name", "drug_fills", "abs_growth", "pct_growth"]].to_string(index=False))

    print("\n  === Largest decline (2024) ===")
    decliners = recent.dropna(subset=["abs_growth"]).nsmallest(5, "abs_growth")
    print(decliners[["brand_name", "drug_fills", "abs_growth", "pct_growth"]].to_string(index=False))

    print("\n  === Highest spending (2024) ===")
    top_spend = recent.nlargest(5, "drug_spending")
    print(top_spend[["brand_name", "drug_spending", "spending_market_share"]].to_string(index=False))

    return dy


def main():
    con = duckdb.connect(DB_PATH, read_only=True)

    market_yearly = task9_market_size(con)
    market_yearly = task10_market_growth(market_yearly)
    drug_yearly = task11_drug_market_share(con, market_yearly)
    therapy_yearly = task12_class_performance(drug_yearly)
    growth_leaders = task13_growth_leaders(drug_yearly)

    con.close()

    # Save outputs
    market_yearly.to_csv("outputs/market_yearly.csv", index=False)
    drug_yearly.to_csv("outputs/drug_yearly.csv", index=False)
    therapy_yearly.to_csv("outputs/therapy_yearly.csv", index=False)

    print(f"\n{'='*60}")
    print("TASKS 9-13 COMPLETE — Market analytics saved to outputs/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
