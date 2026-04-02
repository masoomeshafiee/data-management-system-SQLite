import sqlite3
import streamlit as st

from queries.browse_queries import get_distinct_values

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

