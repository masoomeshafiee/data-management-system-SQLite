
import streamlit as st

def page_header(title: str, subtitle: str | None = None):
    st.title(title)
    if subtitle:
        st.caption(subtitle)

def page_placeholder(title: str):
    page_header(title)
    st.info("We’ll build this page next.")

def require_user_selected() -> int:
    if "current_user_id" not in st.session_state or st.session_state.current_user_id is None:
        st.warning("Select your user name in the sidebar to continue.")
        st.stop()
    return int(st.session_state.current_user_id)