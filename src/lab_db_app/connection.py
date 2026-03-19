import sqlite3
import streamlit as st


DB_PATH = st.secrets.get("DB_PATH", "")

import sqlite3
from contextlib import contextmanager

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn

@contextmanager
def get_connection(db_path: str):
    conn = get_conn(db_path)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise
