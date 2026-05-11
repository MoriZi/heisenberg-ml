"""
Database connection utilities.

Loads credentials from .env at the project root and provides
both raw psycopg2 connections and SQLAlchemy engines.
"""

import os
from pathlib import Path
from urllib.parse import quote_plus

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Walk up from this file to find the project-root .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def get_connection():
    """Return a raw psycopg2 connection using env-var credentials."""
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def get_engine():
    """Return a SQLAlchemy engine (useful for pandas read_sql)."""
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ["DB_PASSWORD"]
    url = f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{dbname}"
    return create_engine(url)
