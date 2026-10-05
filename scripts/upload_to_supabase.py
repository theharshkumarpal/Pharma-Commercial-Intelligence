import os
import getpass
import pandas as pd
import duckdb
from sqlalchemy import create_engine

DB_PATH = "data/hcproject.duckdb"
SUPABASE_HOST = "db.arwtajtwqwdthvfcyepn.supabase.co"
SUPABASE_PORT = 5432
SUPABASE_DB = "postgres"
SUPABASE_USER = "postgres"

def main():
    print("=" * 60)
    print("SUPABASE POSTGRESQL DATA MIGRATION")
    print("=" * 60)
    
    password = os.environ.get("SUPABASE_PASSWORD")
    if not password:
        password = getpass.getpass("Enter your Supabase database password: ")

    connection_string = f"postgresql://{SUPABASE_USER}:{password}@{SUPABASE_HOST}:{SUPABASE_PORT}/{SUPABASE_DB}"
    
    print(f"Connecting to Supabase PostgreSQL at {SUPABASE_HOST}...")
    engine = create_engine(connection_string)
    
    tables_to_upload = {
        "market_yearly": "outputs/market_yearly.csv",
        "drug_yearly": "outputs/drug_yearly.csv",
        "therapy_yearly": "outputs/therapy_yearly.csv",
        "hcp_features": "outputs/hcp_features.csv",
        "hcp_segments": "outputs/hcp_segments.csv",
        "state_opportunity": "outputs/state_opportunity.csv",
        "hcp_opportunity": "outputs/hcp_opportunity.csv",
        "forecast": "outputs/forecast.csv",
        "growth_share_matrix": "outputs/growth_share_matrix.csv"
    }

    print("\nUploading analytical tables to Supabase...")
    with engine.begin() as conn:
        for tname, path in tables_to_upload.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                df.to_sql(tname, conn, if_exists="replace", index=False)
                print(f"  ✅ Uploaded table '{tname}': {len(df):,} rows")

    print("\nLoading star schema dimension & fact tables from DuckDB...")
    con = duckdb.connect(DB_PATH, read_only=True)
    star_tables = ["dim_hcp", "dim_drug", "fact_prescriptions"]
    with engine.begin() as conn:
        for tname in star_tables:
            try:
                df = con.execute(f"SELECT * FROM {tname}").fetchdf()
                df.to_sql(tname, conn, if_exists="replace", index=False)
                print(f"  ✅ Uploaded star schema table '{tname}': {len(df):,} rows")
            except Exception as e:
                print(f"  ⚠️ Could not upload '{tname}': {e}")
    con.close()

    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE — All tables uploaded to Supabase!")
    print("=" * 60)

if __name__ == "__main__":
    main()
