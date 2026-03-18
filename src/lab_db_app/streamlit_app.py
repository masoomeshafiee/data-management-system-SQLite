
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
    db_path = os.environ.get("DB_PATH", st.secrets.get("DB_PATH", "../db/Reyes_lab_data.db"))
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
        options=["Browse/Search", "Create Experiment", "Attach Files", "Edit", "Admin","Data Intake"],
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


def page_browse_v1(conn: sqlite3.Connection):

    page_header("Browse / Search", "Filter and open experiments")

    with st.form("search_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            organism = st.text_input("Organism")
            protein = st.text_input("Protein")
        with col2:
            condition = st.text_input("Condition")
            capture_type = st.text_input("Capture type")
        with col3:
            user_name = st.text_input("User")
            is_valid = st.selectbox("Is valid", ["", "1", "0"])

        submitted = st.form_submit_button("Search")

    filters = {}
    if organism.strip():
        filters["organism"] = organism.strip()
    if protein.strip():
        filters["protein"] = protein.strip()
    if condition.strip():
        filters["condition"] = condition.strip()
    if capture_type.strip():
        filters["capture_type"] = capture_type.strip()
    if user_name.strip():
        filters["user_name"] = user_name.strip()
    if is_valid:
        filters["is_valid"] = is_valid

    df = search_experiments(conn, filters=filters, limit=200)
    st.dataframe(df, use_container_width=True, hide_index=True)

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

def get_saved_searches() -> dict:
    return st.session_state.setdefault("saved_searches", {})

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

def page_browse(conn: sqlite3.Connection):
    from queries_v2 import (
        FIELD_REGISTRY,
        get_filterable_fields_for_target,
        get_selectable_fields_for_target,
        get_default_columns_for_target,
        group_fields_by_section,
        search_table,
        get_experiment_metadata,
        get_raw_files_for_experiment,
        get_tracking_files_for_experiment,
        get_masks_for_experiment,
        get_analysis_files_for_experiment,
        get_results_for_experiment
    )

    page_header("Browse / Search", "Search experiments and related tables")

    TARGET_OPTIONS = [
        "Experiment",
        "User",
        "RawFiles",
        "TrackingFiles",
        "Masks",
        "AnalysisFiles",
        "AnalysisResults",
    ]

    # -----------------------------
    # Target table
    # -----------------------------
    target_table = st.selectbox("Search target", TARGET_OPTIONS, index=0, key="browse_target_table")

    # Key namespace per target to avoid stale widget collisions
    prefix = f"browse_{target_table}"

    # -----------------------------
    # Saved searches
    # -----------------------------
    saved_searches = get_saved_searches()

    st.subheader("Saved searches")
    colA, colB = st.columns([2, 1])

    with colA:
        preset_name = st.selectbox(
            "Load saved search",
            options=[""] + list(saved_searches.keys()),
            index=0,
            key=f"{prefix}_preset_name",
        )

    with colB:
        if preset_name and st.button("Load preset", key=f"{prefix}_load_preset"):
            preset = saved_searches[preset_name]

            # only load if preset matches target table
            if preset.get("target_table") != target_table:
                st.warning(
                    f"This preset was saved for target table '{preset.get('target_table')}', "
                    f"not '{target_table}'."
                )
            else:
                st.session_state[f"{prefix}_selected_filters"] = preset.get("selected_filters", [])
                st.session_state[f"{prefix}_filters"] = preset.get("filters", {})
                st.session_state[f"{prefix}_selected_columns"] = preset.get("selected_columns", [])
                st.session_state[f"{prefix}_limit"] = preset.get("limit", 200)
                st.rerun()

    # -----------------------------
    # Field metadata
    # -----------------------------
    filter_fields = get_filterable_fields_for_target(target_table)
    selectable_fields = get_selectable_fields_for_target(target_table)
    default_columns = get_default_columns_for_target(target_table)

    grouped_filter_fields = group_fields_by_section(filter_fields)
    grouped_selectable_fields = group_fields_by_section(selectable_fields)

    # -----------------------------
    # Selected filters
    # -----------------------------
    if f"{prefix}_selected_filters" not in st.session_state:
        st.session_state[f"{prefix}_selected_filters"] = []

    st.subheader("Filters")

    selected_filter_aliases = []
    for section, fields in grouped_filter_fields.items():
        with st.expander(
            section.replace("_", " ").title(),
            expanded=(section in {"experiment", "sample", "microscopy", "user"}),
        ):
            choices = [f.alias for f in fields]
            selected = st.multiselect(
                f"{section.replace('_', ' ').title()} filters",
                options=choices,
                default=[
                    x for x in st.session_state.get(f"{prefix}_selected_filters", [])
                    if x in choices
                ],
                format_func=lambda a: FIELD_REGISTRY[a].output_label,
                key=f"{prefix}_filter_select_{section}",
            )
            selected_filter_aliases.extend(selected)

    # persist selected filters
    st.session_state[f"{prefix}_selected_filters"] = selected_filter_aliases

    # -----------------------------
    # Render filter value widgets immediately
    # -----------------------------
    filters = {}
    saved_filter_values = st.session_state.get(f"{prefix}_filters", {})

    cols = st.columns(3)
    for i, alias in enumerate(selected_filter_aliases):
        field = FIELD_REGISTRY[alias]
        col = cols[i % 3]
        with col:
            filters = render_filter_widget_with_defaults(
                conn=conn,
                filters=filters,
                field=field,
                default_value=saved_filter_values.get(alias),
                key_prefix=prefix,
            )

    # persist current filter values
    st.session_state[f"{prefix}_filters"] = filters

    # -----------------------------
    # Columns to display
    # -----------------------------
    st.subheader("Columns to display")

    if f"{prefix}_selected_columns" not in st.session_state:
        st.session_state[f"{prefix}_selected_columns"] = default_columns.copy()

    selected_columns = []
    for section, fields in grouped_selectable_fields.items():
        with st.expander(
            section.replace("_", " ").title(),
            expanded=(section in {"experiment", "sample", "user"}),
        ):
            choices = [f.alias for f in fields]
            defaults = [
                c for c in st.session_state.get(f"{prefix}_selected_columns", default_columns)
                if c in choices
            ]

            cols_for_section = st.multiselect(
                f"{section.replace('_', ' ').title()} columns",
                options=choices,
                default=defaults,
                format_func=lambda a: FIELD_REGISTRY[a].output_label,
                key=f"{prefix}_column_select_{section}",
            )
            selected_columns.extend(cols_for_section)

    # dedupe while preserving order
    selected_columns = list(dict.fromkeys(selected_columns))
    st.session_state[f"{prefix}_selected_columns"] = selected_columns

    # -----------------------------
    # Limit + preset save
    # -----------------------------
    limit = st.number_input(
        "Result limit",
        min_value=1,
        max_value=5000,
        value=st.session_state.get(f"{prefix}_limit", 200),
        step=50,
        key=f"{prefix}_limit_widget",
    )
    st.session_state[f"{prefix}_limit"] = int(limit)

    save_preset_name = st.text_input("Save current search as preset (name)", key=f"{prefix}_save_preset_name")
    save_preset = st.checkbox("Save this search preset", value=False, key=f"{prefix}_save_preset_flag")

    # -----------------------------
    # Search button
    # -----------------------------
    if st.button("Search", type="primary", key=f"{prefix}_search_button"):
        if not selected_columns:
            st.warning("Choose at least one result column.")
        else:
            if save_preset and save_preset_name.strip():
                saved_searches[save_preset_name.strip()] = {
                    "target_table": target_table,
                    "selected_filters": selected_filter_aliases,
                    "filters": filters,
                    "selected_columns": selected_columns,
                    "limit": int(limit),
                }
                st.session_state["saved_searches"] = saved_searches

            try:
                df = search_table(
                    conn=conn,
                    main_table=target_table,
                    filters=filters,
                    requested_columns=selected_columns,
                    limit=int(limit),
                )
                st.session_state[f"{prefix}_results"] = df
            except Exception as e:
                st.error(f"Search failed: {e}")

    # -----------------------------
    # Show last results
    # -----------------------------
    df = st.session_state.get(f"{prefix}_results")

    if df is not None:
        st.subheader("Results")
        st.dataframe(df, use_container_width=True, hide_index=True)

        if not df.empty:
            st.download_button(
                "Download results (CSV)",
                data=df.to_csv(index=False),
                file_name=f"{target_table.lower()}_search_results.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
            )

            render_result_detail_ui(target_table, df)

        else:
            st.info("No results found.")

    # -----------------------------
    # Experiment detail section
    # -----------------------------
    selected_experiment_id = st.session_state.get("selected_experiment_id")
    if target_table == "Experiment" and selected_experiment_id:
        st.divider()
        st.subheader(f"Experiment details: {selected_experiment_id}")

        try:
            meta_df = get_experiment_metadata(conn, selected_experiment_id)
            st.markdown("**Metadata**")
            st.dataframe(meta_df, use_container_width=True, hide_index=True)

            raw_df = get_raw_files_for_experiment(conn, selected_experiment_id)
            tracking_df = get_tracking_files_for_experiment(conn, selected_experiment_id)
            mask_df = get_masks_for_experiment(conn, selected_experiment_id)
            analysis_df = get_analysis_files_for_experiment(conn, selected_experiment_id)
            result = get_results_for_experiment(conn, selected_experiment_id)

            tabs = st.tabs(["Raw files", "Tracking files", "Masks", "Analysis files", "Results"])

            with tabs[0]:
                st.dataframe(raw_df, use_container_width=True, hide_index=True)
            with tabs[1]:
                st.dataframe(tracking_df, use_container_width=True, hide_index=True)
            with tabs[2]:
                st.dataframe(mask_df, use_container_width=True, hide_index=True)
            with tabs[3]:
                st.dataframe(analysis_df, use_container_width=True, hide_index=True)
            with tabs[4]:
                st.dataframe(result, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Could not load experiment details: {e}")

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
    if page == "Data Intake":
        st.switch_page("pages/data_intake.py")
    elif page == "Create Experiment":
        page_create_experiment(conn)
    elif page == "Browse/Search":
        page_browse(conn)
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

if __name__ == "__main__":
    main()