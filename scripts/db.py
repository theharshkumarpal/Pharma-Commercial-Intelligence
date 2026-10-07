import os
import sys
from sqlalchemy import create_engine

SUPABASE_HOST = os.environ.get("SUPABASE_HOST", "aws-0-ap-northeast-2.pooler.supabase.com")
SUPABASE_PORT = int(os.environ.get("SUPABASE_PORT", "6543"))
SUPABASE_DB = os.environ.get("SUPABASE_DB", "postgres")
SUPABASE_USER = os.environ.get("SUPABASE_USER", "postgres.arwtajtwqwdthvfcyepn")

def get_db_engine():
    """Get SQLAlchemy engine for Supabase PostgreSQL database."""
    password = os.environ.get("SUPABASE_PASSWORD")
    if not password:
        # Check if local fallback file exists or prompt if needed
        env_file = ".env"
        if os.path.exists(env_file):
            with open(env_file) as f:
                for line in f:
                    if line.startswith("SUPABASE_PASSWORD="):
                        password = line.strip().split("=", 1)[1]
                        break
    if not password:
        raise ValueError(
            "SUPABASE_PASSWORD environment variable is missing. "
            "Please set SUPABASE_PASSWORD in your environment or .env file."
        )
    connection_string = f"postgresql://{SUPABASE_USER}:{password}@{SUPABASE_HOST}:{SUPABASE_PORT}/{SUPABASE_DB}"
    return create_engine(connection_string)
