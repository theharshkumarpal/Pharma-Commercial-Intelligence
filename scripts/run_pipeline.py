"""
Master Pipeline Runner — Executes all 37 tasks in sequence.

Run with:  python scripts/run_pipeline.py
"""
import subprocess
import sys
import time

SCRIPTS = [
    ("01", "scripts/01_data_dictionary.py",              "Task 2 — Data Dictionary"),
    ("02", "scripts/02_clean_combine_load.py",            "Tasks 3-8 — Clean, Combine, Map, Load, Validate"),
    ("03", "scripts/03_market_analytics.py",              "Tasks 9-13 — Market Analytics"),
    ("04", "scripts/04_hcp_analytics.py",                 "Tasks 14-22 — HCP Analytics & Segmentation"),
    ("05", "scripts/05_adoption_geography_opportunity.py", "Tasks 23-28 — Adoption, Geography, Opportunity"),
    ("06", "scripts/06_forecasting.py",                   "Tasks 29-34 — Forecasting"),
    ("07", "scripts/07_statistical_analysis.py",          "Tasks 35-37 — Statistics & Final Tables"),
    ("08", "scripts/08_quality_check.py",                 "Part L — Final Quality Check"),
]

def main():
    print("=" * 70)
    print("PHARMA COMMERCIAL INTELLIGENCE PIPELINE")
    print("HCP Segmentation, Drug Adoption & Market Forecasting")
    print("=" * 70)
    start = time.time()

    for num, script, desc in SCRIPTS:
        print(f"\n{'#'*70}")
        print(f"# [{num}] {desc}")
        print(f"# Script: {script}")
        print(f"{'#'*70}")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n❌ FAILED: {script} (exit code {result.returncode})")
            print(f"   Fix the issue and re-run.")
            sys.exit(1)
        print(f"\n  ✅ {desc} completed in {elapsed:.1f}s")

    total = time.time() - start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — Total time: {total:.1f}s")
    print(f"{'='*70}")
    print(f"\nOutputs saved to:")
    print(f"  data/hcproject.duckdb   — Analytical database")
    print(f"  outputs/                — CSV/JSON analytical tables")
    print(f"  docs/                   — Data dictionary & market definition")
    print(f"\n✅ Ready for Power BI.")

if __name__ == "__main__":
    main()
