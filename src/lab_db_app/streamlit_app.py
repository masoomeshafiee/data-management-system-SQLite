
from __future__ import annotations

import streamlit as st
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

import os
import sys
import sqlite3
import pandas as pd

from queries_v2 import list_users, search_experiments, get_experiment_metadata
from config import load_config


TARGET_OPTIONS = [
        "Experiment",
        "Condition",
        "Organism",
        "Protein",
        "StrainOrCellLine",
        "CaptureSetting",
        "User",
        "RawFiles",
        "TrackingFiles",
        "Masks",
        "AnalysisFiles",
        "Results",
    ]
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

def create_user(conn: sqlite3.Connection, user_name: str, last_name: str | None, email: str | None) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO User (user_name, last_name, email)
        VALUES (?, ?, ?)
        """,
        (
            user_name.strip(),
            last_name.strip() if last_name else None,
            email.strip() if email else None,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)

# -----------------------------
# data access
# -----------------------------
# we will use the queries module.



# -----------------------------
# Main App
# -----------------------------
def sidebar(conn: sqlite3.Connection):
    st.sidebar.header("Lab Metadata DB")

    # User selection
    users_df = list_users(conn)
    if users_df.empty:
        st.session_state.current_user_id = None
        st.sidebar.warning("No users found in DB.")
        st.sidebar.info("Create the first user to initialize the app.")
        page = st.sidebar.radio(
            label="Go to",
            options=["Setup / Create First User", "Data Intake"],
            index=0,
        )
        return page

    options = users_df.apply(lambda r: f'{r["user_name"]} {r["last_name"]} ({r["email"]})', axis=1).tolist()
    id_map = dict(zip(options, users_df["id"].tolist()))

    default_opt = options[0]
    selected = st.sidebar.selectbox("Current user", options, index=0)
    st.session_state.current_user_id = id_map[selected]

    st.sidebar.divider()
    st.sidebar.write("Navigation")
    page = st.sidebar.radio(
        label="Go to",
        options=["Home", "Browse/Search","Data Intake", "Create Experiment", "Attach Files", "Update data", "Delete data","Edit", "Admin"],
        index=0,
    )
    return page

def page_setup_first_user(conn: sqlite3.Connection):
    page_header("Setup / Create First User", "Initialize the app with the first lab user")

    st.warning("No users exist in the database yet. Create the first user to continue.")

    with st.form("create_first_user_form", clear_on_submit=False):
        user_name = st.text_input("User name *")
        last_name = st.text_input("Last name")
        email = st.text_input("Email")

        submitted = st.form_submit_button("Create first user")

    if submitted:
        if not user_name.strip():
            st.error("User name is required.")
            st.stop()

        try:
            user_id = create_user(conn, user_name, last_name, email)
            st.session_state.current_user_id = user_id
            st.success(f"User created successfully (id={user_id}). Please rerun or continue.")
            st.rerun()
        except sqlite3.IntegrityError as e:
            st.error(f"Could not create user: {e}")


# ==============================
# helpers for browse/search page
# ==============================
def get_distinct_values(conn: sqlite3.Connection, table: str, column: str) -> list[str]:
    query = f"""
        SELECT DISTINCT {column}
        FROM {table}
        WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS TEXT)) <> ''
        ORDER BY {column} COLLATE NOCASE
    """
    cur = conn.execute(query)
    return [row[0] for row in cur.fetchall()]


def render_filter_widget_with_defaults(
    conn: sqlite3.Connection,
    filters: dict,
    field,
    default_value=None,
    key_prefix: str = "",
) -> dict:
    label = field.output_label
    key = f"{key_prefix}_filter_value_{field.alias}"

    def init_state(default):
        if key not in st.session_state and default not in (None, ""):
            st.session_state[key] = default

    # Special cases
    if field.alias == "is_valid":
        init_state(default_value if default_value is not None else "")
        value = st.selectbox(label, options=["", "1", "0"], key=key)
        if value != "":
            filters[field.alias] = value
        return filters

    if field.data_type == "date":
        init_state(default_value if default_value is not None else "")
        value = st.text_input(label, key=key, placeholder="YYYY-MM-DD")
        if value.strip():
            filters[field.alias] = value.strip()
        return filters

    dropdown_fields = {
        "organism", "protein", "strain", "condition",
        "capture_type", "user_name", "email",
        "raw_file_type", "tracking_file_type", "mask_type", "mask_file_type", "analysis_file_type",
    }

    if field.alias in dropdown_fields:
        try:
            values = get_distinct_values(conn, field.table, field.column)
            init_state(default_value if default_value is not None else "")
            value = st.selectbox(label, options=[""] + values, key=key)
            if value != "":
                filters[field.alias] = value
        except Exception:
            init_state(default_value if default_value is not None else "")
            value = st.text_input(label, key=key)
            if value.strip():
                filters[field.alias] = value.strip()
        return filters

    if field.data_type == "int":
        init_state(str(default_value) if default_value is not None else "")
        value = st.text_input(label, key=key)
        if value.strip():
            try:
                filters[field.alias] = int(value)
            except ValueError:
                st.warning(f"{label} must be an integer.")
        return filters

    if field.data_type == "float":
        init_state(str(default_value) if default_value is not None else "")
        value = st.text_input(label, key=key)
        if value.strip():
            try:
                filters[field.alias] = float(value)
            except ValueError:
                st.warning(f"{label} must be numeric.")
        return filters

    init_state(default_value if default_value is not None else "")
    value = st.text_input(label, key=key)
    if value.strip():
        filters[field.alias] = value.strip()
    return filters


def render_result_detail_ui(target_table: str, df: pd.DataFrame):
    if target_table != "Experiment":
        return

    if "experiment_id" not in df.columns or df.empty:
        return

    experiment_ids = df["experiment_id"].dropna().tolist()
    if not experiment_ids:
        return

    selected_experiment_id = st.selectbox(
        "Open experiment details",
        options=experiment_ids,
        index=0,
        key="experiment_detail_selectbox",
    )

    if st.button("Show experiment details", key="experiment_detail_button"):
        st.session_state["selected_experiment_id"] = int(selected_experiment_id)
        st.rerun()
# ==============================
# Main page functions
# ==============================


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

    if page == "Setup / Create First User":
        page_setup_first_user(conn)
    if page == "Home":
        page_header("Lab Metadata DB", "Welcome")
        st.write("Select a page from the sidebar.")
    elif page == "Data Intake":
        st.switch_page("pages/data_intake.py")
    elif page == "Create Experiment":
        page_create_experiment(conn)
    elif page == "Update data":
        page_placeholder("Update Experiment")
    elif page == "Delete data":
        page_placeholder("Delete Experiment")
    elif page == "Attach Files":
        page_placeholder("Attach Files")
    elif page == "Edit":
        page_placeholder("Edit")
    elif page == "Admin":
        page_placeholder("Admin")
    elif page == "Browse/Search":
        st.switch_page("pages/browse_search.py")

if __name__ == "__main__":
    main()