import sqlite3
import pandas as pd
import streamlit as st

from config import FIELD_REGISTRY, load_config
from db.connection import get_connection
from browse_search import render_filter_widget_with_defaults 
from queries.queries_utils import group_fields_by_section, get_filterable_fields_for_target
from streamlit_app import page_header

from services.qc_service import (
        run_default_qc_suite,
        qc_experiments_missing_files,
        qc_experiments_missing_metadata,
        qc_duplicate_experiments,
        qc_experiments_with_analysis_but_no_results,
        qc_analysis_files_without_results,
        qc_results_without_analysis_files,
        qc_incomplete_linked_entities,
    )

# ----------------------------------------------------
#           Helpers
# ----------------------------------------------------
def render_dynamic_filter_panel(
    conn: sqlite3.Connection,
    *,
    target_table: str,
    prefix: str,
):


    st.session_state.setdefault(f"{prefix}_selected_filters", [])
    st.session_state.setdefault(f"{prefix}_filters", {})

    filter_fields = get_filterable_fields_for_target(target_table)
    grouped_filter_fields = group_fields_by_section(filter_fields)

    st.subheader("Scope filters")

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

    st.session_state[f"{prefix}_selected_filters"] = selected_filter_aliases

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

    st.session_state[f"{prefix}_filters"] = filters
    return filters

# ------------------------------------------------------
# Main page
# -----------------------------------------------------

def page_quality_control(conn: sqlite3.Connection):

    page_header("Quality Control", "Run integrity and completeness checks on the database")

    prefix = "qc_experiment_scope"

    # -----------------------------
    # Pre-run QC scope filters
    # -----------------------------
    experiment_filters = render_dynamic_filter_panel(
        conn,
        target_table="Experiment",
        prefix=prefix,
    )

    st.subheader("QC checks")

    run_default = st.checkbox("Run default QC suite", value=True)

    if not run_default:
        colA, colB = st.columns(2)

        with colA:
            run_missing_files = st.checkbox("Missing expected files", value=True)
            run_missing_metadata = st.checkbox("Missing metadata", value=True)
            run_duplicates = st.checkbox("Duplicate experiments", value=True)

        with colB:
            run_analysis_no_results = st.checkbox("Experiments with analysis but no results", value=True)
            run_orphan_analysis = st.checkbox("Analysis files without results", value=True)
            run_orphan_results = st.checkbox("Results without analysis files", value=True)

        st.markdown("**Optional linked-entity QC**")
        run_raw_without_tracking = st.checkbox(
            "Experiments with RawFiles but no TrackingFiles",
            value=False,
        )
    else:
        run_missing_files = run_missing_metadata = run_duplicates = True
        run_analysis_no_results = run_orphan_analysis = run_orphan_results = True
        run_raw_without_tracking = False

    limit_per_check = st.number_input(
        "Max rows per QC check",
        min_value=10,
        max_value=5000,
        value=500,
        step=50,
        key="qc_limit_per_check",
    )

    run_qc_clicked = st.button("Run QC", type="primary")

    if run_qc_clicked:
        outputs = []

        try:
            if run_default:
                qc_df = run_default_qc_suite(
                    conn,
                    experiment_filters=experiment_filters,
                    limit_per_check=int(limit_per_check),
                )
            else:
                if run_missing_files:
                    outputs.append(
                        qc_experiments_missing_files(
                            conn,
                            file_types=("raw", "tracking", "mask", "analysis"),
                            filters=experiment_filters,
                            limit=int(limit_per_check),
                        )
                    )

                if run_missing_metadata:
                    outputs.append(
                        qc_experiments_missing_metadata(
                            conn,
                            required_fields=["organism", "protein", "condition", "capture_type", "date", "replicate"],
                            filters=experiment_filters,
                            mode="any",
                            limit=int(limit_per_check),
                        )
                    )

                if run_duplicates:
                    outputs.append(
                        qc_duplicate_experiments(
                            conn,
                            filters=experiment_filters,
                        )
                    )

                if run_analysis_no_results:
                    outputs.append(
                        qc_experiments_with_analysis_but_no_results(
                            conn,
                            filters=experiment_filters,
                            limit=int(limit_per_check),
                        )
                    )

                if run_orphan_analysis:
                    outputs.append(
                        qc_analysis_files_without_results(
                            conn,
                            limit=int(limit_per_check),
                        )
                    )

                if run_orphan_results:
                    outputs.append(
                        qc_results_without_analysis_files(
                            conn,
                            limit=int(limit_per_check),
                        )
                    )

                if run_raw_without_tracking:
                    outputs.append(
                        qc_incomplete_linked_entities(
                            conn,
                            base_table="Experiment",
                            present_entity=("RawFiles", "experiment_id"),
                            missing_entity=("TrackingFiles", "experiment_id"),
                            filters=experiment_filters,
                            limit=int(limit_per_check),
                            summary="Experiment has raw files but no tracking files.",
                        )
                    )

                non_empty = [df for df in outputs if not df.empty]
                if non_empty:
                    qc_df = pd.concat(non_empty, ignore_index=True)
                else:
                    qc_df = pd.DataFrame(
                        columns=[
                            "entity_type",
                            "entity_id",
                            "issue_category",
                            "severity",
                            "issue_summary",
                            "issue_details",
                        ]
                    )

            st.session_state["qc_results_df"] = qc_df

        except Exception as e:
            st.error(f"QC run failed: {e}")

    qc_df = st.session_state.get("qc_results_df")

    if qc_df is not None:
        st.divider()
        st.subheader("QC results")

        if qc_df.empty:
            st.success("No QC issues found for the selected checks.")
            return

        # -----------------------------
        # Post-run issue filters
        # -----------------------------
        st.markdown("**Filter QC results**")
        colF1, colF2, colF3 = st.columns(3)

        with colF1:
            issue_categories = ["All"] + sorted(qc_df["issue_category"].dropna().unique().tolist())
            selected_category = st.selectbox("Issue category", issue_categories, index=0)

        with colF2:
            severities = ["All"] + sorted(qc_df["severity"].dropna().unique().tolist())
            selected_severity = st.selectbox("Severity", severities, index=0)

        with colF3:
            entity_types = ["All"] + sorted(qc_df["entity_type"].dropna().unique().tolist())
            selected_entity_type = st.selectbox("Entity type", entity_types, index=0)

        filtered_qc_df = qc_df.copy()

        if selected_category != "All":
            filtered_qc_df = filtered_qc_df[filtered_qc_df["issue_category"] == selected_category]
        if selected_severity != "All":
            filtered_qc_df = filtered_qc_df[filtered_qc_df["severity"] == selected_severity]
        if selected_entity_type != "All":
            filtered_qc_df = filtered_qc_df[filtered_qc_df["entity_type"] == selected_entity_type]

        # Summary
        colS1, colS2 = st.columns(2)
        with colS1:
            st.markdown("**Issue counts by category**")
            summary_by_category = (
                filtered_qc_df.groupby("issue_category")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.dataframe(summary_by_category, use_container_width=True, hide_index=True)

        with colS2:
            st.markdown("**Issue counts by severity**")
            summary_by_severity = (
                filtered_qc_df.groupby("severity")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.dataframe(summary_by_severity, use_container_width=True, hide_index=True)

        st.markdown("**Detailed issues**")
        st.dataframe(filtered_qc_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download QC issues (CSV)",
            data=filtered_qc_df.to_csv(index=False),
            file_name="qc_issues.csv",
            mime="text/csv",
        )

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Quality Control", layout="wide")
st.title("Data Quality Control")
st.markdown(
    "Navigate through the page to performe Completeness/Workflow QC, Duplicate/consistency checks,"
    "invalid experiments detection, and Coverage summaries"
)
cfg = load_config()

with get_connection(cfg.db_path) as conn:
    page_quality_control(conn)
