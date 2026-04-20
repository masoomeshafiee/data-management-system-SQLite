
from __future__ import annotations

import streamlit as st
import sqlite3

from config import load_config
from db.connection import get_conn
from app.ui.layout import page_header, page_placeholder
from queries.browse_queries import list_users
from queries.insert_queries import create_user


# -----------------------------
# Main App Helpers
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
        options=["Home", "Browse/Search","Data Intake", "Quality Control", "Create Experiment", "Attach Files", "Update data", "Delete data","Edit", "Admin"],
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
# Main page functions
# ==============================


def page_create_experiment(conn: sqlite3.Connection):
    page_header("Create Experiment", "Register a new experiment record")

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



def main():
    
    st.set_page_config(page_title="Lab Metadata DB", layout="wide")
    cfg = load_config()

    conn = get_conn(cfg.db_path)
    page = sidebar(conn) # to be able to list the users in the db

    if page == "Setup / Create First User":
        page_setup_first_user(conn) # independent of data intake, we can add user to the db, so we dont rely on one round of data insertion to be able to access a user
    if page == "Home":
        page_header("Lab Metadata DB", "Welcome")
        st.write("Select a page from the sidebar.")
    elif page == "Data Intake":
        st.switch_page("pages/data_intake.py")
    elif page == "Analytics":
        page_placeholder("Analytics")
    elif page == "Admin":
        page_placeholder("Admin")
    elif page == "Browse/Search":
        st.switch_page("pages/browse_search.py")
    elif page == "Quality Control":
        st.swithc_page("pages/quality_control.py")

if __name__ == "__main__":
    main()