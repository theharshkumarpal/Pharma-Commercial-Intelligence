"""
TASK 2 — Build Data Dictionary
Document every field from the CMS Part D Prescribers by Provider & Drug dataset.
"""
import json
import os

DATA_DICTIONARY = {
    "dataset": "Medicare Part D Prescribers - by Provider and Drug",
    "source": "CMS (Centers for Medicare & Medicaid Services)",
    "years_used": "2019–2024 (finalized annual data)",
    "fields": [
        {
            "field": "Year",
            "type": "integer",
            "description": "Calendar year of the data (added during processing; not in original CMS file).",
            "cms_definition": "Derived from the annual file name / release year."
        },
        {
            "field": "Prscrbr_NPI",
            "type": "string (10 digits)",
            "description": "National Provider Identifier — unique 10-digit ID for the prescribing provider.",
            "cms_definition": "National Provider Identifier for the performing provider on the Part D claim."
        },
        {
            "field": "Prscrbr_Type",
            "type": "string",
            "description": "Provider specialty (e.g. Internal Medicine, Endocrinology, Family Practice).",
            "cms_definition": "Derived from Medicare claims or NPPES taxonomy mapping when claims-based specialty is not available."
        },
        {
            "field": "Prscrbr_State_Abrvtn",
            "type": "string (2 chars)",
            "description": "U.S. state abbreviation of the provider's practice location.",
            "cms_definition": "The state where the provider is located, as reported in NPPES."
        },
        {
            "field": "Brnd_Name",
            "type": "string",
            "description": "Brand (trade) name of the drug as it appears on the claim.",
            "cms_definition": "The trade name under which the drug is sold."
        },
        {
            "field": "Gnrc_Name",
            "type": "string",
            "description": "Generic (active ingredient) name of the drug.",
            "cms_definition": "Chemical ingredient name of the drug product."
        },
        {
            "field": "Tot_Clms",
            "type": "numeric (float)",
            "description": "Total number of Medicare Part D claims including original prescriptions and refills.",
            "cms_definition": "Number of Medicare Part D claims, including original prescriptions and refills. Suppressed where < 11.",
            "suppression_note": "Blank/null values may represent CMS suppression (< 11 claims). Do NOT replace with zero."
        },
        {
            "field": "Tot_30day_Fills",
            "type": "numeric (float)",
            "description": "Standardized 30-day fill count. Normalizes prescriptions of varying day-supply to a common 30-day unit.",
            "cms_definition": "Total standardized 30-day fills. Calculated by dividing total day supply by 30.",
            "suppression_note": "Subject to same suppression rules as Tot_Clms."
        },
        {
            "field": "Tot_Drug_Cst",
            "type": "numeric (float, USD)",
            "description": "Aggregate drug cost for this provider–drug combination in the given year.",
            "cms_definition": "Aggregate cost paid for all associated claims including ingredient cost, dispensing fee, sales tax, and any applicable vaccine administration fees.",
            "suppression_note": "Subject to same suppression rules as Tot_Clms."
        },
        {
            "field": "Tot_Day_Suply",
            "type": "integer",
            "description": "Total days supply for all claims of this drug by this provider.",
            "cms_definition": "Aggregate number of day's supply for all claims.",
            "suppression_note": "Subject to same suppression rules as Tot_Clms."
        },
        {
            "field": "Tot_Benes",
            "type": "numeric (float)",
            "description": "Total number of unique Medicare Part D beneficiaries who received this drug from this provider.",
            "cms_definition": "Total number of unique Medicare Part D beneficiaries with at least one claim. Suppressed where < 11.",
            "suppression_note": "Subject to same suppression rules as Tot_Clms."
        }
    ],
    "suppression_policy": {
        "description": "CMS suppresses data when the number of claims or beneficiaries is fewer than 11 to protect patient privacy.",
        "handling": "Suppressed values are stored as NULL/NaN. A boolean 'is_suppressed' flag is created during cleaning. These records are NEVER replaced with zero as that would bias the analysis."
    },
    "key_relationships": {
        "primary_grain": "Year + NPI + Drug (Brand/Generic combination)",
        "provider_dimension": "NPI → Specialty, State",
        "drug_dimension": "Brand/Generic → RxCUI → Ingredient → Therapeutic Class → Diabetes Category"
    }
}


def main():
    os.makedirs("docs", exist_ok=True)
    outpath = "docs/data_dictionary.json"
    with open(outpath, "w") as f:
        json.dump(DATA_DICTIONARY, f, indent=2)
    print(f"Data dictionary written to {outpath}")
    
    # Also write a readable markdown version
    md_path = "docs/data_dictionary.md"
    with open(md_path, "w") as f:
        f.write("# CMS Part D Prescribers by Provider & Drug — Data Dictionary\n\n")
        f.write(f"**Source:** {DATA_DICTIONARY['source']}  \n")
        f.write(f"**Years:** {DATA_DICTIONARY['years_used']}  \n\n")
        f.write("## Fields\n\n")
        f.write("| Field | Type | CMS Definition | Suppression |\n")
        f.write("|-------|------|----------------|-------------|\n")
        for fld in DATA_DICTIONARY["fields"]:
            sup = fld.get("suppression_note", "N/A")
            f.write(f"| `{fld['field']}` | {fld['type']} | {fld['cms_definition']} | {sup} |\n")
        f.write(f"\n## Suppression Policy\n\n{DATA_DICTIONARY['suppression_policy']['description']}\n\n")
        f.write(f"**Handling:** {DATA_DICTIONARY['suppression_policy']['handling']}\n\n")
        f.write(f"## Key Relationships\n\n")
        for k, v in DATA_DICTIONARY["key_relationships"].items():
            f.write(f"- **{k}:** {v}\n")
    print(f"Markdown data dictionary written to {md_path}")


if __name__ == "__main__":
    main()
