
import sqlite3
import streamlit as st
import pandas as pd
from config import load_config, FIELD_REGISTRY, TARGET_OPTIONS

from queries.queries_utils import (
        get_filterable_fields_for_target,
        get_selectable_fields_for_target,
        get_default_columns_for_target,
        group_fields_by_section,
        get_id_field_for_target
        )

from queries.browse_queries import (
        search_table,
        get_experiment_metadata,
        get_raw_files_for_experiment,
        get_tracking_files_for_experiment,
        get_masks_for_experiment,
        get_analysis_files_for_experiment,
        get_results_for_experiment
    )

from queries.save_searche import (
    list_saved_searches,
    get_saved_search_by_id,
    create_saved_search,
    update_saved_search,
    delete_saved_search,
)
from db.connection import get_connection
from app.ui.layout import page_header, require_user_selected
from app.ui.filter_widgets import render_filter_widget_with_defaults

from app.ui.action_panels import (
    render_selectable_table,
    get_selected_rows,
    render_action_buttons,
)
from services.delete_service import (
    preview_delete_experiments,
    execute_delete_experiments,
    preview_delete_analysis_files,
    execute_delete_analysis_files,
    preview_delete_raw_files,
    execute_delete_raw_files,
    preview_delete_tracking_files,
    execute_delete_tracking_files,
    preview_delete_masks,
    execute_delete_masks,
    preview_delete_results,
    execute_delete_results,
    preview_delete_reference_entities,
    execute_delete_reference_entities,
)



# ======================================================
# Helpers
# ======================================================


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

def _preview_delete_for_target(
    conn: sqlite3.Connection,
    *,
    target_table: str,
    selected_rows: pd.DataFrame,
):
    id_column = get_id_field_for_target(target_table)
    if id_column is None:
        raise ValueError(f"Delete is not supported for target table '{target_table}'.")

    if id_column not in selected_rows.columns:
        raise ValueError(
            f"Selected rows do not contain the required id column '{id_column}' for target '{target_table}'."
        )

    ids = selected_rows[id_column].dropna().tolist()

    if target_table == "Experiment":
        return preview_delete_experiments(conn, ids)

    if target_table == "AnalysisFiles":
        return preview_delete_analysis_files(conn, ids)

    if target_table == "RawFiles":
        return preview_delete_raw_files(conn, ids)

    if target_table == "TrackingFiles":
        return preview_delete_tracking_files(conn, ids)

    if target_table == "Masks":
        return preview_delete_masks(conn, ids)

    if target_table == "Results":
        return preview_delete_results(conn, ids)

    if target_table in {"Organism", "Protein", "StrainOrCellLine", "Condition", "CaptureSetting", "User"}:
        return preview_delete_reference_entities(
            conn,
            parent_table=target_table,
            parent_ids=ids,
        )

    raise ValueError(f"Delete is not supported for target table '{target_table}'.")

def _execute_delete_for_target(
    conn: sqlite3.Connection,
    *,
    target_table: str,
    selected_ids: list[int],
):
    """
    Route delete execution by target_table.
    Returns a DeleteExecutionResult from the delete service.
    """
    if target_table == "Experiment":
        return execute_delete_experiments(conn, selected_ids)

    if target_table == "AnalysisFiles":
        return execute_delete_analysis_files(conn, selected_ids)

    if target_table == "RawFiles":
        return execute_delete_raw_files(conn, selected_ids)

    if target_table == "TrackingFiles":
        return execute_delete_tracking_files(conn, selected_ids)

    if target_table == "Masks":
        return execute_delete_masks(conn, selected_ids)

    if target_table == "Results":
        return execute_delete_results(conn, selected_ids)

    if target_table in {"Organism", "Protein", "StrainOrCellLine", "Condition", "CaptureSetting", "User"}:
        return execute_delete_reference_entities(
            conn,
            parent_table=target_table,
            parent_ids=selected_ids,
        )

    raise ValueError(f"Delete is not supported for target table '{target_table}'.")
# ----------------------------------
# Main function
# ----------------------------------
def page_browse(conn: sqlite3.Connection):
    

    page_header("Browse / Search", "Search experiments and related tables")


    # -----------------------------
    # Target table
    # -----------------------------
    target_table = st.selectbox("Search target", TARGET_OPTIONS, index=0, key="browse_target_table")

    if target_table != "Experiment":
        st.session_state.pop("selected_experiment_id", None)

    # Key namespace per target to avoid stale widget collisions
    prefix = f"browse_{target_table}"

    st.session_state.setdefault(f"{prefix}_selected_filters", [])
    st.session_state.setdefault(f"{prefix}_filters", {})
    st.session_state.setdefault(f"{prefix}_selected_columns", get_default_columns_for_target(target_table))
    st.session_state.setdefault(f"{prefix}_limit", 200)

    # -----------------------------
    # Saved searches
    # -----------------------------

    
    current_user_id = require_user_selected()

    st.subheader("Saved searches")

    presets_df = list_saved_searches(
        conn,
        user_id=current_user_id,
        target_table=target_table,
        include_shared=True,
    )

    preset_choices = [None] + [
        {
            "id": int(row["id"]),
            "label": f'{row["name"]}{" [shared]" if int(row["is_shared"]) == 1 else ""} - {row["user_name"]}',
        }
        for _, row in presets_df.iterrows()
    ]

    colA, colB, colC = st.columns([2, 1, 1])

    with colA:
        preset_choice = st.selectbox(
            "Load saved search",
            options=preset_choices,
            index=0,
            format_func=lambda x: "" if x is None else x["label"],
            key=f"{prefix}_preset_name",
        )

    with colB:
        if preset_choice is not None and st.button("Load preset", key=f"{prefix}_load_preset"):
            preset_id = preset_choice["id"]
            preset = get_saved_search_by_id(conn, preset_id)

            if preset is None:
                st.error("Preset not found.")
            elif preset.get("target_table") != target_table:
                st.warning(
                    f"This preset was saved for target table '{preset.get('target_table')}', "
                    f"not '{target_table}'."
                )
            else:
                st.session_state[f"{prefix}_selected_filters"] = preset.get("selected_filters", [])
                st.session_state[f"{prefix}_filters"] = preset.get("filters", {})
                st.session_state[f"{prefix}_selected_columns"] = preset.get("selected_columns", [])
                st.session_state[f"{prefix}_limit"] = preset.get("limit", 200)
                st.session_state[f"{prefix}_loaded_preset_id"] = preset["id"]
                st.session_state[f"{prefix}_loaded_preset_name"] = preset["name"]
                st.session_state[f"{prefix}_loaded_preset_is_shared"] = preset["is_shared"]
                st.rerun()

    with colC:
        if preset_choice is not None and st.button("Delete preset", key=f"{prefix}_delete_preset"):
            preset_id = preset_choice["id"]
            preset = get_saved_search_by_id(conn, preset_id)

            if preset is None:
                st.error("Preset not found.")
            elif int(preset["created_by_user_id"]) != int(current_user_id):
                st.error("You can only delete your own saved searches.")
            else:
                delete_saved_search(conn, preset_id)
                st.success("Preset deleted.")
                st.session_state.pop(f"{prefix}_loaded_preset_id", None)
                st.session_state.pop(f"{prefix}_loaded_preset_name", None)
                st.session_state.pop(f"{prefix}_loaded_preset_is_shared", None)
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

    st.subheader("Save current search")

    loaded_preset_id = st.session_state.get(f"{prefix}_loaded_preset_id")
    loaded_preset_name = st.session_state.get(f"{prefix}_loaded_preset_name", "")
    loaded_preset_is_shared = st.session_state.get(f"{prefix}_loaded_preset_is_shared", False)

    save_preset_name = st.text_input(
        "Preset name",
        value=loaded_preset_name,
        key=f"{prefix}_save_preset_name",
    )
    is_shared = st.checkbox(
        "Shared preset",
        value=bool(loaded_preset_is_shared),
        key=f"{prefix}_save_preset_shared",
        help="Shared presets are visible to other users.",
    )

    col_save1, col_save2 = st.columns(2)
    with col_save1:
        save_as_new = st.checkbox("Save as new preset", value=(loaded_preset_id is None), key=f"{prefix}_save_as_new")
    with col_save2:
        save_preset_clicked = st.button("Save preset", key=f"{prefix}_save_preset_button")
    # -----------------------------
    # Search button
    # -----------------------------
    if st.button("Search", type="primary", key=f"{prefix}_search_button"):
        if not selected_columns:
            st.warning("Choose at least one result column.")
        else:
            try:
                id_field_alias = get_id_field_for_target(target_table)

                query_columns = selected_columns.copy()
                if id_field_alias and id_field_alias not in query_columns:
                    query_columns.append(id_field_alias)
                df = search_table(
                    conn=conn,
                    main_table=target_table,
                    filters=filters,
                    requested_columns=query_columns,
                    limit=int(limit),
                )
                st.session_state[f"{prefix}_results"] = df
            except Exception as e:
                st.error(f"Search failed: {e}")
    if save_preset_clicked:
        if not save_preset_name.strip():
            st.warning("Enter a preset name.")
        else:
            try:
                if loaded_preset_id is not None and not save_as_new:
                    # update existing preset only if owned by current user
                    existing = get_saved_search_by_id(conn, loaded_preset_id)
                    if existing is None:
                        st.error("Loaded preset no longer exists.")
                    elif int(existing["created_by_user_id"]) != int(current_user_id):
                        st.error("You can only update your own saved searches.")
                    else:
                        update_saved_search(
                            conn,
                            saved_search_id=loaded_preset_id,
                            name=save_preset_name.strip(),
                            target_table=target_table,
                            filters=filters,
                            selected_filters=selected_filter_aliases,
                            selected_columns=selected_columns,
                            result_limit=int(limit),
                            is_shared=is_shared,
                        )
                        st.success("Preset updated.")
                else:
                    create_saved_search(
                        conn,
                        name=save_preset_name.strip(),
                        target_table=target_table,
                        filters=filters,
                        selected_filters=selected_filter_aliases,
                        selected_columns=selected_columns,
                        result_limit=int(limit),
                        created_by_user_id=current_user_id,
                        is_shared=is_shared,
                    )
                    st.success("Preset saved.")

                st.rerun()

            except sqlite3.IntegrityError:
                st.error("A preset with this name already exists for your account.")
            except Exception as e:
                st.error(f"Could not save preset: {e}")

    # -----------------------------
    # Show last results
    # -----------------------------
    df = st.session_state.get(f"{prefix}_results")

    if df is not None:
        st.subheader("Results")

        if df.empty:
            st.info("No results found.")
        else:
            # main results download
            st.download_button(
                "Download results (CSV)",
                data=df.to_csv(index=False),
                file_name=f"{target_table.lower()}_search_results.csv",
                mime="text/csv",
                key=f"{prefix}_download_csv",
            )

            # selectable table for actions
            edited_df = render_selectable_table(
                df,
                key=f"{prefix}_results_editor",
            )
            selected_rows = get_selected_rows(edited_df)

            st.caption(f"{len(selected_rows)} row(s) selected.")

            delete_clicked, mark_invalid_clicked, export_clicked = render_action_buttons(
                key_prefix=f"{prefix}_actions"
            )

            # Export selected
            if export_clicked:
                if selected_rows.empty:
                    st.warning("Please select at least one row to export.")
                else:
                    st.download_button(
                        "Download selected rows (CSV)",
                        data=selected_rows.to_csv(index=False),
                        file_name=f"{target_table.lower()}_selected_rows.csv",
                        mime="text/csv",
                        key=f"{prefix}_download_selected_csv",
                    )

            # Mark invalid
            if mark_invalid_clicked:
                if selected_rows.empty:
                    st.warning("Please select at least one row to mark invalid.")
                elif target_table != "Experiment":
                    st.warning("Mark invalid is currently supported only for Experiment results.")
                else:
                    st.info("Mark invalid is not implemented yet.")

            # Delete selected -> preview
            if delete_clicked:
                if selected_rows.empty:
                    st.warning("Please select at least one row to delete.")
                else:
                    try:
                        delete_preview = _preview_delete_for_target(
                            conn,
                            target_table=target_table,
                            selected_rows=selected_rows,
                        )
                        st.session_state[f"{prefix}_delete_preview"] = delete_preview
                        st.session_state[f"{prefix}_delete_target_table"] = target_table
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not prepare delete preview: {e}")

            # delete preview panel
            delete_preview = st.session_state.get(f"{prefix}_delete_preview")
            if delete_preview is not None:
                st.divider()
                st.subheader("Delete preview")
                st.warning(delete_preview.message)

                if not delete_preview.summary_df.empty:
                    st.markdown("**Summary of affected records**")
                    st.dataframe(delete_preview.summary_df, use_container_width=True, hide_index=True)

                with st.expander("Show details", expanded=False):
                    for name, detail_df in delete_preview.details.items():
                        st.markdown(f"**{name.replace('_', ' ').title()}**")
                        if detail_df.empty:
                            st.caption("No rows.")
                        else:
                            st.dataframe(detail_df, use_container_width=True, hide_index=True)

                col_confirm, col_cancel = st.columns(2)

                with col_confirm:
                    confirm_delete = st.button(
                        "Confirm delete",
                        type="primary",
                        key=f"{prefix}_confirm_delete",
                    )

                with col_cancel:
                    cancel_delete = st.button(
                        "Cancel",
                        key=f"{prefix}_cancel_delete",
                    )

                if cancel_delete:
                    st.session_state.pop(f"{prefix}_delete_preview", None)
                    st.rerun()

                if confirm_delete:
                    if confirm_delete:
                        try:
                            delete_target_table = st.session_state.get(f"{prefix}_delete_target_table", target_table)

                            selected_ids = (
                                delete_preview.selected_ids
                                if hasattr(delete_preview, "selected_ids")
                                else delete_preview["selected_ids"]
                            )

                            delete_result = _execute_delete_for_target(
                                conn,
                                target_table=delete_target_table,
                                selected_ids=selected_ids,
                            )

                            st.session_state[f"{prefix}_delete_result"] = delete_result
                            st.session_state.pop(f"{prefix}_delete_preview", None)
                            st.session_state.pop(f"{prefix}_delete_target_table", None)
                            st.session_state.pop(f"{prefix}_results", None)
                            st.session_state.pop("selected_experiment_id", None)
                            st.rerun()

                        except Exception as e:
                            st.error(f"Delete failed: {e}")

            # delete result message
            delete_result = st.session_state.get(f"{prefix}_delete_result")
            if delete_result is not None:
                if delete_result.status == "ok":
                    st.success(delete_result.message)
                else:
                    st.error(delete_result.message)

            render_result_detail_ui(target_table, df)

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
            result_df = get_results_for_experiment(conn, selected_experiment_id)

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
                st.dataframe(result_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Could not load experiment details: {e}")

# ----------------------------
# Streamlit UI
# ----------------------------


st.set_page_config(page_title="Browse/Search Lab Database", layout="wide")
st.title("Browse/Search Lab Database")
st.markdown(
    "Use the controls bellow to search for experiments, files, and results. "
    "Click on an experiment in the results to see more details."
)
cfg = load_config()

with get_connection(cfg.db_path) as conn:
    page_browse(conn)


