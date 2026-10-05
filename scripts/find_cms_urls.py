import requests
import json

# CMS API endpoint for searching datasets
api_url = "https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug"

r = requests.get(api_url)
print("Status:", r.status_code)

# Search CMS datasets API endpoint for "Medicare Part D Prescribers - by Provider and Drug"
search_url = "https://data.cms.gov/data-api/v1/dataset"
r2 = requests.get(search_url)
print("Dataset API status:", r2.status_code)
datasets = r2.json() if r2.status_code == 200 else []

print(f"Total datasets listed: {len(datasets)}")
for ds in datasets:
    title = ds.get('title', '')
    if 'Part D Prescribers' in title or 'Prescribers' in title:
        print(f"Title: {title} | Modified: {ds.get('modified')} | ID: {ds.get('identifier')}")
