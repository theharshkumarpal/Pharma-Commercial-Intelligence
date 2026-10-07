# Pharma Commercial Intelligence: HCP Segmentation, Drug Adoption & Market Forecasting

A comprehensive commercial analytics engine built for pharmaceutical opportunity assessment, HCP prescribing segmentation, drug adoption tracking, geographic market analysis, and time-series forecasting using Medicare Part D dataset (2019–2024).

---

## 📌 Project Overview & Scope

### Primary Business Question
> *Using Medicare Part D prescribing data (2019–2024), which diabetes therapies, HCP segments, and geographic markets represent the strongest potential commercial opportunities, and how is prescribing likely to evolve?*

This project is structured around key pharmaceutical commercial analytics capabilities:
- **Opportunity Assessment**: Strategic evaluation of therapeutic class shifts ($GLP-1$, $SGLT2$, $DPP-4$, $Insulin$).
- **Customer Segmentation**: Provider behavioral clustering using K-Means ($K=4$).
- **Product Adoption**: HCP-level prescribing intensity trends.
- **Geographic Mapping**: State-level market share vs. YoY growth opportunity matrix.
- **Forecasting**: 2025–2026 trajectory modeling with scenario analysis (Base, Upside, Downside) using Exponential Smoothing, ARIMA, and Naive backtesting models.
- **Statistical Rigor**: Kruskal-Wallis, Mann-Whitney U, and Spearman correlation hypothesis testing.

---

## 🛠️ Repository Architecture

```text
HCProject/
├── data/                       # Database & tabular files
│   ├── raw/                    # CMS Part D raw source CSVs
│   └── processed/              # Cleaned dimension & fact tables
├── docs/                       # Project documentation
│   ├── DATA_DICTIONARY.md      # Detailed column definitions & datatypes
│   └── DIABETES_MARKET_DEFINITION.md # Therapeutic class mapping & inclusion criteria
├── outputs/                    # Processed analytical outputs & JSON reports
│   ├── market_yearly.csv
│   ├── drug_yearly.csv
│   ├── therapy_yearly.csv
│   ├── hcp_features.csv
│   ├── hcp_segments.csv
│   ├── state_opportunity.csv
│   ├── hcp_opportunity.csv
│   ├── forecast.csv
│   ├── statistical_analysis.json
│   └── quality_check_report.json
├── scripts/                    # End-to-end Python pipeline scripts
│   ├── 00_generate_raw_data.py # Synthetic CMS data generator (2019-2024)
│   ├── 01_data_dictionary.py   # Schema & market definition generator
│   ├── 02_clean_combine_load.py # Cleaning, suppression handling, Supabase PostgreSQL star schema
│   ├── 03_market_analytics.py  # Market size, growth, and therapy share calculations
│   ├── 04_hcp_analytics.py     # HCP feature engineering & K-Means clustering
│   ├── 05_adoption_geography_opportunity.py # Opportunity score & sensitivity matrix
│   ├── 06_forecasting.py       # Time series backtesting & 2-year scenario forecasting
│   ├── 07_statistical_analysis.py # Non-parametric tests & Supabase PostgreSQL table staging
│   ├── 08_quality_check.py     # 20-point automated quality check
│   └── run_pipeline.py         # One-click Master Pipeline Orchestrator
└── README.md
```

---

## 📊 Key Business Findings & Recommendations

### 💡 Findings
1. **GLP-1 Market Dominance**: GLP-1 receptor agonists reached **31.4% market share** in 2024 with a **29.9% YoY growth rate**, outperforming all other diabetes classes.
2. **Declining Insulin Prescribing**: Insulin total fills exhibited flat/negative growth (**-0.1% YoY in 2024**), reflecting earlier-line therapy adoption.
3. **Prescriber Clustering**: Identified 4 distinct HCP archetypes:
   - *High-Volume Core Prescribers* (Avg ~42k fills, 73.4% brand share)
   - *Diversified Prescribers* (Avg ~8.7k fills, 73.4% brand share)
   - *Moderate-Volume High-Growth Prescribers*
   - *Low-Volume / Low-Growth Prescribers*
4. **Robust Opportunity Scoring**: Provider composite opportunity scores showed **80% average rank stability** across sensitivity weight variations.

### 🎯 Strategic Commercial Recommendations
1. **Target Core & Emerging Segments**: Prioritize detailing of GLP-1 and SGLT2 therapies to *High-Volume Core* and *Diversified Prescribers* to maximize immediate ROI.
2. **Geographic Focus**: Direct field force expansion toward states situated in the *Emerging Opportunity Quadrant* (high market growth, lower current brand penetration).
3. **Therapy Switching Campaign**: Target prescribers showing declining DPP-4 and Sulfonylurea volumes for GLP-1/SGLT2 therapy transition programs.

---

## ⚡ Quick Start & Pipeline Execution

### Prerequisites
- Python 3.10+
- Virtual environment (`venv`)

### Execution
Run the full analytical engine with a single command:
```bash
python scripts/run_pipeline.py
```
This executes all 8 sub-modules, populates `hcproject.duckdb`, exports analytical CSVs, and runs the automated 20-point quality check.

---

## ⚙️ Quality Assurance

The analytical engine includes an automated quality assurance auditor ([`scripts/08_quality_check.py`](file:///Users/vandy/Projects/HCProject/scripts/08_quality_check.py)) that validates:
- **Data Integrity**: 2019–2024 completeness, key uniqueness, and suppression compliance.
- **Analytics Completeness**: Market metrics, segmentation stability, backtesting, and sensitivity bounds.
- **Business Guardrails**: Minimum finding counts, recommendation mapping, and observational statistical labeling (no unsupported causal assertions).

> **Status**: `20/20 Checks Passed (100% Complete)`

