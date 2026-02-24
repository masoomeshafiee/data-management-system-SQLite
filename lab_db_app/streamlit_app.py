
from __future__ import annotations

import streamlit as st
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import os
import sys
import sqlite3
import pandas as pd


# -----------------------------------
# Configuration
# -----------------------------------

@dataclass
class AppConfig:
    app_name: str = "Reyes Lab Database App"
    db_path: str = "app_database.db"
    debug_mode: bool = False    
    # Add other configuration parameters as needed

def load_config() -> AppConfig:
    # Put your DB path in an env var on the lab PC
    app_name = os.environ.get("DB_APP_NAME", "Reyes Lab Database App")
    db_path = os.environ.get("REYES_LAB_DB_PATH", "../db/Reyes_lab_data.db")
    debug_mode = os.environ.get("DB_APP_DEBUG_MODE", "False").lower() in ("true", "1", "t")

    return AppConfig(app_name=app_name, db_path=db_path, debug_mode=debug_mode)


# -----------------------------
# DB connection helpers
# -----------------------------

def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # Critical pragmas for your use case (multi-user, SQLite file)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")  # ms
    return conn

@contextmanager
def transaction(conn: sqlite3.Connection):
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise

# -----------------------------
# UI helpers
# -----------------------------
def page_header(title: str, subtitle: str | None = None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def require_user_selected() -> int:
    if "current_user_id" not in st.session_state or st.session_state.current_user_id is None:
        st.warning("Select your user name in the sidebar to continue.")
        st.stop()
    return int(st.session_state.current_user_id)

# -----------------------------
# data access
# -----------------------------

def list_users(conn: sqlite3.Connection) -> pd.DataFrame:
    # return query.list_users(conn)
    df = pd.read_sql_query("SELECT id, user_name, last_name, email FROM User ORDER BY user_name;", conn)
    return df

def search_experiments(conn: sqlite3.Connection, filters: Dict[str, Any]) -> pd.DataFrame:
    # return query.search_experiments(conn, filters)
    # Minimal example: show last 200 experiments
    sql = """
    SELECT
        e.id,
        e.date,
        e.replicate,
        e.comment,
        u.user_name,
        o.organism_name,
        p.protein_name,
        s.strain_name,
        c.condition_name,
        cs.capture_type
    FROM Experiment e
    LEFT JOIN User u ON u.id = e.user_id
    LEFT JOIN Organism o ON o.id = e.organism_id
    LEFT JOIN Protein p ON p.id = e.protein_id
    LEFT JOIN StrainOrCellLine s ON s.id = e.strain_id
    LEFT JOIN Condition c ON c.id = e.condition_id
    LEFT JOIN CaptureSetting cs ON cs.id = e.capture_setting_id
    ORDER BY e.date DESC
    LIMIT 200;
    """
    return pd.read_sql_query(sql, conn)
# -----------------------------
# Main App
# -----------------------------
def sidebar(conn: sqlite3.Connection):
    st.sidebar.header("Lab Metadata DB")

    # User selection
    users_df = list_users(conn)
    if users_df.empty:
        st.sidebar.error("No users found in DB. Add a user first.")
        return

    options = users_df.apply(lambda r: f'{r["user_name"]} {r["last_name"]} ({r["email"]})', axis=1).tolist()
    id_map = dict(zip(options, users_df["id"].tolist()))

    default_opt = options[0]
    selected = st.sidebar.selectbox("Current user", options, index=0)
    st.session_state.current_user_id = id_map[selected]

    st.sidebar.divider()
    st.sidebar.write("Navigation")
    page = st.sidebar.radio(
        label="Go to",
        options=["Browse/Search", "Create Experiment", "Attach Files", "Edit", "Admin"],
        index=0,
    )
    return page


def page_browse(conn: sqlite3.Connection):
    page_header("Browse / Search", "Filter and open experiments")

    # (v1) Simple view; we’ll add real filters next step
    df = search_experiments(conn, filters={})
    st.dataframe(df, width=True, hide_index=True)

    st.info("Next step: add filters + click-to-open experiment detail view.")




def page_create_experiment(conn: sqlite3.Connection):
    page_header("Create Experiment", "Register a new experiment record")

    user_id = require_user_selected()

    # For v1, use raw IDs; next iteration we’ll add dropdowns for lookup tables
    with st.form("create_experiment_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            organism_id = st.number_input("organism_id", min_value=1, step=1)
            protein_id = st.number_input("protein_id", min_value=1, step=1)
            strain_id = st.number_input("strain_id", min_value=1, step=1)
            condition_id = st.number_input("condition_id", min_value=1, step=1)

        with col2:
            capture_setting_id = st.number_input("capture_setting_id", min_value=1, step=1)
            date = st.text_input("date (YYYY-MM-DD)")
            replicate = st.number_input("replicate", min_value=1, step=1)
            is_valid = st.selectbox("is_valid", ["1", "0"], index=0)

        comment = st.text_area("comment", height=80)
        experiment_path = st.text_input("experiment_path")
        #submitted = st.form_submit_button("Create experiment")

        # if submitted:
        #     payload = dict(
        #         organism_id=int(organism_id),
        #         protein_id=int(protein_id),
        #         strain_id=int(strain_id),
        #         condition_id=int(condition_id),
        #         capture_setting_id=int(capture_setting_id),
        #         user_id=int(user_id),
        #         date=date.strip(),
        #         replicate=int(replicate),
        #         is_valid=is_valid,
        #         comment=comment.strip() if comment else None,
        #         experiment_path=experiment_path.strip() if experiment_path else None,
        #     )
        #     exp_id = create_experiment(conn, payload)
        #     st.success(f"Created Experiment id={exp_id}")

def page_placeholder(title: str):
    page_header(title)
    st.info("We’ll build this page next.")


def main():
    st.set_page_config(page_title="Lab Metadata DB", layout="wide")
    cfg = load_config()

    conn = get_conn(cfg.db_path)
    page = sidebar(conn)

    if page == "Browse/Search":
        page_browse(conn)
    elif page == "Create Experiment":
        page_create_experiment(conn)
    elif page == "Attach Files":
        page_placeholder("Attach Files")
    elif page == "Edit":
        page_placeholder("Edit")
    elif page == "Admin":
        page_placeholder("Admin")


if __name__ == "__main__":
    main()