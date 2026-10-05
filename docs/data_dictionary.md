# CMS Part D Prescribers by Provider & Drug — Data Dictionary

**Source:** CMS (Centers for Medicare & Medicaid Services)  
**Years:** 2019–2024 (finalized annual data)  

## Fields

| Field | Type | CMS Definition | Suppression |
|-------|------|----------------|-------------|
| `Year` | integer | Derived from the annual file name / release year. | N/A |
| `Prscrbr_NPI` | string (10 digits) | National Provider Identifier for the performing provider on the Part D claim. | N/A |
| `Prscrbr_Type` | string | Derived from Medicare claims or NPPES taxonomy mapping when claims-based specialty is not available. | N/A |
| `Prscrbr_State_Abrvtn` | string (2 chars) | The state where the provider is located, as reported in NPPES. | N/A |
| `Brnd_Name` | string | The trade name under which the drug is sold. | N/A |
| `Gnrc_Name` | string | Chemical ingredient name of the drug product. | N/A |
| `Tot_Clms` | numeric (float) | Number of Medicare Part D claims, including original prescriptions and refills. Suppressed where < 11. | Blank/null values may represent CMS suppression (< 11 claims). Do NOT replace with zero. |
| `Tot_30day_Fills` | numeric (float) | Total standardized 30-day fills. Calculated by dividing total day supply by 30. | Subject to same suppression rules as Tot_Clms. |
| `Tot_Drug_Cst` | numeric (float, USD) | Aggregate cost paid for all associated claims including ingredient cost, dispensing fee, sales tax, and any applicable vaccine administration fees. | Subject to same suppression rules as Tot_Clms. |
| `Tot_Day_Suply` | integer | Aggregate number of day's supply for all claims. | Subject to same suppression rules as Tot_Clms. |
| `Tot_Benes` | numeric (float) | Total number of unique Medicare Part D beneficiaries with at least one claim. Suppressed where < 11. | Subject to same suppression rules as Tot_Clms. |

## Suppression Policy

CMS suppresses data when the number of claims or beneficiaries is fewer than 11 to protect patient privacy.

**Handling:** Suppressed values are stored as NULL/NaN. A boolean 'is_suppressed' flag is created during cleaning. These records are NEVER replaced with zero as that would bias the analysis.

## Key Relationships

- **primary_grain:** Year + NPI + Drug (Brand/Generic combination)
- **provider_dimension:** NPI → Specialty, State
- **drug_dimension:** Brand/Generic → RxCUI → Ingredient → Therapeutic Class → Diabetes Category
