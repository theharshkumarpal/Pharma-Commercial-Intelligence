import os
import numpy as np
import pandas as pd

# Set random seed for reproducible realistic data generation
np.random.seed(42)

def generate_raw_cms_data():
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    
    # 1. Master List of Diabetes & Non-Diabetes Drugs in CMS Part D
    drugs_info = [
        # GLP-1 Receptor Agonists
        {"Brnd_Name": "OZEMPIC", "Gnrc_Name": "SEMAGLUTIDE", "class": "GLP-1 receptor agonists", "base_cost": 950.0, "base_claims": 80, "trend": 1.45},
        {"Brnd_Name": "MOUNJARO", "Gnrc_Name": "TIRZEPATIDE", "class": "GLP-1 receptor agonists", "base_cost": 1050.0, "base_claims": 0, "trend": 2.20}, # Introduced ~2022
        {"Brnd_Name": "TRULICITY", "Gnrc_Name": "DULAGLUTIDE", "class": "GLP-1 receptor agonists", "base_cost": 880.0, "base_claims": 110, "trend": 1.12},
        {"Brnd_Name": "VICTOZA", "Gnrc_Name": "LIRAGLUTIDE", "class": "GLP-1 receptor agonists", "base_cost": 820.0, "base_claims": 90, "trend": 0.92},
        {"Brnd_Name": "RYBELSUS", "Gnrc_Name": "SEMAGLUTIDE", "class": "GLP-1 receptor agonists", "base_cost": 920.0, "base_claims": 30, "trend": 1.35},

        # SGLT2 Inhibitors
        {"Brnd_Name": "JARDIANCE", "Gnrc_Name": "EMPAGLIFLOZIN", "class": "SGLT2 inhibitors", "base_cost": 570.0, "base_claims": 140, "trend": 1.22},
        {"Brnd_Name": "FARXIGA", "Gnrc_Name": "DAPAGLIFLOZIN", "class": "SGLT2 inhibitors", "base_cost": 550.0, "base_claims": 130, "trend": 1.20},
        {"Brnd_Name": "INVOKANA", "Gnrc_Name": "CANAGLIFLOZIN", "class": "SGLT2 inhibitors", "base_cost": 530.0, "base_claims": 70, "trend": 0.90},
        {"Brnd_Name": "STEGLATRO", "Gnrc_Name": "ERTUGLIFLOZIN", "class": "SGLT2 inhibitors", "base_cost": 320.0, "base_claims": 20, "trend": 1.02},

        # DPP-4 Inhibitors
        {"Brnd_Name": "JANUVIA", "Gnrc_Name": "SITAGLIPTIN", "class": "DPP-4 inhibitors", "base_cost": 480.0, "base_claims": 160, "trend": 0.93},
        {"Brnd_Name": "TRADJENTA", "Gnrc_Name": "LINAGLIPTIN", "class": "DPP-4 inhibitors", "base_cost": 470.0, "base_claims": 100, "trend": 0.94},
        {"Brnd_Name": "ONGLYZA", "Gnrc_Name": "SAXAGLIPTIN", "class": "DPP-4 inhibitors", "base_cost": 450.0, "base_claims": 40, "trend": 0.85},

        # Insulins
        {"Brnd_Name": "LANTUS", "Gnrc_Name": "INSULIN GLARGINE", "class": "Insulin", "base_cost": 420.0, "base_claims": 220, "trend": 0.98},
        {"Brnd_Name": "HUMALOG", "Gnrc_Name": "INSULIN LISPRO", "class": "Insulin", "base_cost": 380.0, "base_claims": 180, "trend": 0.97},
        {"Brnd_Name": "NOVOLOG", "Gnrc_Name": "INSULIN ASPART", "class": "Insulin", "base_cost": 390.0, "base_claims": 170, "trend": 0.96},
        {"Brnd_Name": "TRESIBA", "Gnrc_Name": "INSULIN DEGLUDEC", "class": "Insulin", "base_cost": 460.0, "base_claims": 90, "trend": 1.05},
        {"Brnd_Name": "TOUJEO", "Gnrc_Name": "INSULIN GLARGINE", "class": "Insulin", "base_cost": 440.0, "base_claims": 80, "trend": 1.02},

        # Biguanides
        {"Brnd_Name": "METFORMIN HYDROCHLORIDE", "Gnrc_Name": "METFORMIN HCl", "class": "Biguanides", "base_cost": 25.0, "base_claims": 450, "trend": 1.01},

        # Sulfonylureas
        {"Brnd_Name": "GLIPIZIDE", "Gnrc_Name": "GLIPIZIDE", "class": "Sulfonylureas", "base_cost": 20.0, "base_claims": 210, "trend": 0.95},
        {"Brnd_Name": "GLIMEPIRIDE", "Gnrc_Name": "GLIMEPIRIDE", "class": "Sulfonylureas", "base_cost": 18.0, "base_claims": 190, "trend": 0.94},

        # Thiazolidinediones
        {"Brnd_Name": "PIOGLITAZONE HYDROCHLORIDE", "Gnrc_Name": "PIOGLITAZONE HCl", "class": "Thiazolidinediones", "base_cost": 35.0, "base_claims": 75, "trend": 0.92},

        # Other Diabetes Therapies
        {"Brnd_Name": "SYMLINPEN 120", "Gnrc_Name": "PRAMLINTIDE ACETATE", "class": "Other diabetes therapies", "base_cost": 600.0, "base_claims": 15, "trend": 0.88},
        {"Brnd_Name": "CYCLOSET", "Gnrc_Name": "BROMOCRIPTINE MESYLATE", "class": "Other diabetes therapies", "base_cost": 310.0, "base_claims": 10, "trend": 0.90}
    ]

    # 2. Provider Cohort (5,000 HCPs)
    num_hcps = 4500
    npis = [str(1000000000 + i) for i in range(num_hcps)]
    
    specialties = [
        "Endocrinology", "Internal Medicine", "Family Practice", 
        "Nurse Practitioner", "Physician Assistant", "Cardiology", "Nephrology"
    ]
    specialty_weights = [0.18, 0.38, 0.28, 0.08, 0.05, 0.02, 0.01]
    
    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "NC", "GA", "MI", "AZ", "TN", "MA", "IN", "VA"]
    state_weights = [0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.05, 0.05]
    
    hcp_specialty = np.random.choice(specialties, size=num_hcps, p=specialty_weights)
    hcp_state = np.random.choice(states, size=num_hcps, p=state_weights)
    
    # Provider adoption tier multiplier (High volume, Medium volume, Low volume)
    hcp_volume_tier = np.random.choice([0.2, 1.0, 3.5, 8.0], size=num_hcps, p=[0.40, 0.35, 0.20, 0.05])

    print("Generating raw CMS Part D datasets for 2019-2024...")

    for year in years:
        year_dir = f"data/raw/{year}"
        os.makedirs(year_dir, exist_ok=True)
        records = []
        
        y_idx = year - 2019

        for i, npi in enumerate(npis):
            spec = hcp_specialty[i]
            st = hcp_state[i]
            vol_mult = hcp_volume_tier[i]

            # Select subset of drugs prescribed by this HCP
            num_drugs = np.random.randint(2, 9)
            if spec == "Endocrinology":
                num_drugs = np.random.randint(5, 14)
            
            selected_drugs = np.random.choice(drugs_info, size=min(num_drugs, len(drugs_info)), replace=False)

            for d in selected_drugs:
                b_claims = d["base_claims"]
                trend = d["trend"]
                
                # Mounjaro introduced in 2022
                if d["Brnd_Name"] == "MOUNJARO" and year < 2022:
                    continue

                # Calculate claim count with exponential growth/decline trend
                expected_claims = b_claims * (trend ** y_idx) * vol_mult * np.random.uniform(0.7, 1.3)
                
                # Endocrinology prescribes more GLP-1 and SGLT2
                if spec == "Endocrinology" and d["class"] in ["GLP-1 receptor agonists", "SGLT2 inhibitors"]:
                    expected_claims *= 2.1

                claims = int(np.round(expected_claims))

                if claims < 1:
                    continue
                
                # CMS suppression logic: Small values (< 11 claims) are suppressed/blanked
                is_suppressed = False
                if claims < 11 and np.random.rand() < 0.85:
                    is_suppressed = True
                
                if is_suppressed:
                    tot_clms = None
                    tot_30day_fills = None
                    tot_drug_cst = None
                    tot_day_suply = None
                    tot_benes = None
                else:
                    tot_clms = float(claims)
                    # 30-day fills is typically ~1.1 to 1.3 x claims
                    tot_30day_fills = float(np.round(tot_clms * np.random.uniform(1.05, 1.25), 1))
                    tot_day_suply = int(tot_30day_fills * 30)
                    cost_per_fill = d["base_cost"] * (1 + 0.04 * y_idx) * np.random.uniform(0.95, 1.05)
                    tot_drug_cst = float(np.round(tot_30day_fills * cost_per_fill, 2))
                    tot_benes = float(max(11, int(np.round(tot_clms / np.random.uniform(2.5, 4.5)))))

                records.append({
                    "Year": year,
                    "Prscrbr_NPI": npi,
                    "Prscrbr_Type": spec,
                    "Prscrbr_State_Abrvtn": st,
                    "Brnd_Name": d["Brnd_Name"],
                    "Gnrc_Name": d["Gnrc_Name"],
                    "Tot_Clms": tot_clms,
                    "Tot_30day_Fills": tot_30day_fills,
                    "Tot_Drug_Cst": tot_drug_cst,
                    "Tot_Day_Suply": tot_day_suply,
                    "Tot_Benes": tot_benes
                })

        # Add a small fraction of duplicate records to test cleaning (TASK 3)
        df_year = pd.DataFrame(records)
        dups = df_year.sample(frac=0.002, random_state=year)
        df_year = pd.concat([df_year, dups], ignore_index=True)

        filepath = os.path.join(year_dir, f"Medicare_Part_D_Prescribers_by_Provider_and_Drug_{year}.csv")
        df_year.to_csv(filepath, index=False)
        print(f"Saved {year} dataset to {filepath} ({len(df_year):,} rows)")

if __name__ == "__main__":
    generate_raw_cms_data()
