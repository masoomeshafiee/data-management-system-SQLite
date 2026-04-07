import pandas as pd
import streamlit as st


def add_selection_column(
    df: pd.DataFrame,
    *,
    selection_col: str = "_selected",
    default: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    if selection_col not in out.columns:
        out.insert(0, selection_col, default)
    return out


def render_selectable_table(
    df: pd.DataFrame,
    *,
    key: str,
    selection_col: str = "_selected",
    use_container_width: bool = True,
    hide_index: bool = True,
) -> pd.DataFrame:
    editable_df = add_selection_column(df, selection_col=selection_col)

    edited_df = st.data_editor(
        editable_df,
        key=key,
        use_container_width=use_container_width,
        hide_index=hide_index,
        disabled=[c for c in editable_df.columns if c != selection_col],
        column_config={
            selection_col: st.column_config.CheckboxColumn(
                "Select",
                help="Select row for action",
                default=False,
            )
        },
    )
    return edited_df


def get_selected_rows(
    edited_df: pd.DataFrame,
    *,
    selection_col: str = "_selected",
) -> pd.DataFrame:
    if edited_df.empty or selection_col not in edited_df.columns:
        return pd.DataFrame(columns=edited_df.columns if not edited_df.empty else [])
    return edited_df[edited_df[selection_col]].drop(columns=[selection_col], errors="ignore")


def render_action_buttons(
    *,
    key_prefix: str,
    delete_label: str = "Delete selected",
    invalid_label: str = "Mark invalid",
    export_label: str = "Export selected",
) -> tuple[bool, bool, bool]:
    col1, col2, col3 = st.columns(3)

    with col1:
        delete_clicked = st.button(delete_label, key=f"{key_prefix}_delete")
    with col2:
        invalid_clicked = st.button(invalid_label, key=f"{key_prefix}_mark_invalid")
    with col3:
        export_clicked = st.button(export_label, key=f"{key_prefix}_export")

    return delete_clicked, invalid_clicked, export_clicked