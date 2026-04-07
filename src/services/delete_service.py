from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import pandas as pd
import sqlite3

from config import REFERENCE_PARENT_TABLES
from queries import delete_queries


@dataclass(frozen=True)
class DeletePreviewResult:
    status: str
    message: str
    target_entity: str
    selected_ids: list[int]
    summary_df: pd.DataFrame
    details: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class DeleteExecutionResult:
    status: str
    message: str
    target_entity: str
    selected_ids: list[int]
    deleted_counts: dict[str, int]


def _normalize_ids(ids: Iterable[int | str]) -> list[int]:
    unique_ids: list[int] = []
    seen: set[int] = set()
    for value in ids:
        normalized = int(value)
        if normalized not in seen:
            seen.add(normalized)
            unique_ids.append(normalized)
    return unique_ids


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _validate_non_empty_ids(ids: Iterable[int | str], *, entity_name: str) -> list[int]:
    normalized = _normalize_ids(ids)
    if not normalized:
        raise ValueError(f"No {entity_name} ids were provided.")
    return normalized


def _sum_deleted_counts(counts: dict[str, int]) -> int:
    return sum(v for v in counts.values() if isinstance(v, int))


def extract_ids_from_dataframe(df: pd.DataFrame, *, id_column: str) -> list[int]:
    if df.empty:
        return []
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame.")
    return _normalize_ids(df[id_column].tolist())


def extract_experiment_ids_from_qc_issues(issues_df: pd.DataFrame) -> list[int]:
    if issues_df.empty:
        return []

    required = {"entity_type", "entity_id"}
    missing = required - set(issues_df.columns)
    if missing:
        raise ValueError(f"QC issues dataframe is missing required columns: {sorted(missing)}")

    experiment_rows = issues_df[issues_df["entity_type"].astype(str).str.lower() == "experiment"]
    if experiment_rows.empty:
        return []

    return _normalize_ids(experiment_rows["entity_id"].dropna().tolist())


# =========================================================
# Experiment deletion
# =========================================================

def preview_delete_experiments(
    conn: sqlite3.Connection,
    experiment_ids: Iterable[int | str],
) -> DeletePreviewResult:
    try:
        normalized_ids = _validate_non_empty_ids(experiment_ids, entity_name="experiment")
    except ValueError as exc:
        return DeletePreviewResult(
            status="invalid_selection",
            message=str(exc),
            target_entity="Experiment",
            selected_ids=[],
            summary_df=_empty_df([
                "experiment_count",
                "raw_file_count",
                "tracking_file_count",
                "mask_count",
                "experiment_analysis_links_removed",
                "linked_analysis_file_count",
                "analysis_files_deleted_count",
                "analysis_files_preserved_count",
                "result_analysis_links_removed",
                "linked_result_count",
                "results_deleted_count",
                "results_preserved_count",
            ]),
            details={},
        )

    summary_df = delete_queries.preview_experiment_delete_summary(conn, normalized_ids)
    details = delete_queries.preview_experiment_delete_details(conn, normalized_ids)

    experiments_df = details.get("experiments", pd.DataFrame())
    found_ids = (
        _normalize_ids(experiments_df["id"].tolist())
        if not experiments_df.empty and "id" in experiments_df.columns
        else []
    )
    missing_ids = sorted(set(normalized_ids) - set(found_ids))

    if not found_ids:
        return DeletePreviewResult(
            status="not_found",
            message="None of the selected experiments were found.",
            target_entity="Experiment",
            selected_ids=[],
            summary_df=summary_df,
            details=details,
        )

    if missing_ids:
        return DeletePreviewResult(
            status="partial",
            message=f"Preview generated for {len(found_ids)} experiment(s). Missing ids: {missing_ids}",
            target_entity="Experiment",
            selected_ids=found_ids,
            summary_df=summary_df,
            details=details,
        )

    return DeletePreviewResult(
        status="ok",
        message=f"Preview generated for {len(found_ids)} experiment(s).",
        target_entity="Experiment",
        selected_ids=found_ids,
        summary_df=summary_df,
        details=details,
    )


def preview_delete_experiments_from_filters(
    conn: sqlite3.Connection,
    *,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 500,
) -> DeletePreviewResult:
    experiment_df = delete_queries.preview_experiments_by_filters(
        conn,
        filters=filters,
        limit=limit,
    )

    if experiment_df.empty:
        return DeletePreviewResult(
            status="no_data",
            message="No experiments found for the selected filters.",
            target_entity="Experiment",
            selected_ids=[],
            summary_df=_empty_df([
                "experiment_count",
                "raw_file_count",
                "tracking_file_count",
                "mask_count",
                "experiment_analysis_links_removed",
                "linked_analysis_file_count",
                "analysis_files_deleted_count",
                "analysis_files_preserved_count",
                "result_analysis_links_removed",
                "linked_result_count",
                "results_deleted_count",
                "results_preserved_count",
            ]),
            details={},
        )

    experiment_ids = delete_queries.select_experiment_ids_from_dataframe(experiment_df, id_column="experiment_id")
    return preview_delete_experiments(conn, experiment_ids)


def preview_delete_experiments_from_qc_issues(
    conn: sqlite3.Connection,
    issues_df: pd.DataFrame,
) -> DeletePreviewResult:
    experiment_ids = extract_experiment_ids_from_qc_issues(issues_df)
    if not experiment_ids:
        return DeletePreviewResult(
            status="invalid_selection",
            message="No experiment ids could be extracted from the selected QC issues.",
            target_entity="Experiment",
            selected_ids=[],
            summary_df=_empty_df([
                "experiment_count",
                "raw_file_count",
                "tracking_file_count",
                "mask_count",
                "experiment_analysis_links_removed",
                "linked_analysis_file_count",
                "analysis_files_deleted_count",
                "analysis_files_preserved_count",
                "result_analysis_links_removed",
                "linked_result_count",
                "results_deleted_count",
                "results_preserved_count",
            ]),
            details={},
        )
    return preview_delete_experiments(conn, experiment_ids)


def execute_delete_experiments(
    conn: sqlite3.Connection,
    experiment_ids: Iterable[int | str],
) -> DeleteExecutionResult:
    preview = preview_delete_experiments(conn, experiment_ids)

    if preview.status in {"invalid_selection", "not_found"}:
        return DeleteExecutionResult(
            status=preview.status,
            message=preview.message,
            target_entity="Experiment",
            selected_ids=preview.selected_ids,
            deleted_counts={},
        )

    ids_to_delete = preview.selected_ids
    if not ids_to_delete:
        return DeleteExecutionResult(
            status="invalid_selection",
            message="No valid experiment ids were available for deletion.",
            target_entity="Experiment",
            selected_ids=[],
            deleted_counts={},
        )

    try:
        with conn:
            deleted_counts = delete_queries.delete_experiments_cascade(conn, ids_to_delete)

        return DeleteExecutionResult(
            status="ok",
            message=(
                f"Deleted {deleted_counts.get('experiments_deleted', 0)} experiment(s), "
                f"{deleted_counts.get('analysis_files_deleted', 0)} orphaned analysis file(s), "
                f"and {deleted_counts.get('results_deleted', 0)} orphaned result(s)."
            ),
            target_entity="Experiment",
            selected_ids=ids_to_delete,
            deleted_counts=deleted_counts,
        )
    except Exception as exc:
        return DeleteExecutionResult(
            status="error",
            message=f"Failed to delete experiments: {exc}",
            target_entity="Experiment",
            selected_ids=ids_to_delete,
            deleted_counts={},
        )


# =========================================================
# Analysis file deletion
# =========================================================

def preview_delete_analysis_files(
    conn: sqlite3.Connection,
    analysis_file_ids: Iterable[int | str],
) -> DeletePreviewResult:
    try:
        normalized_ids = _validate_non_empty_ids(analysis_file_ids, entity_name="analysis file")
    except ValueError as exc:
        return DeletePreviewResult(
            status="invalid_selection",
            message=str(exc),
            target_entity="AnalysisFiles",
            selected_ids=[],
            summary_df=_empty_df([
                "analysis_file_count",
                "experiment_analysis_links_removed",
                "result_analysis_links_removed",
                "linked_result_count",
                "results_deleted_count",
                "results_preserved_count",
            ]),
            details={},
        )

    summary_df = delete_queries.preview_analysis_file_delete_summary(conn, normalized_ids)
    details = {
        "analysis_files": delete_queries.preview_delete_rows_by_ids(
            conn,
            table="AnalysisFiles",
            ids=normalized_ids,
            id_column="id",
        )
    }

    found_ids = (
        _normalize_ids(details["analysis_files"]["id"].tolist())
        if not details["analysis_files"].empty and "id" in details["analysis_files"].columns
        else []
    )
    missing_ids = sorted(set(normalized_ids) - set(found_ids))

    if not found_ids:
        return DeletePreviewResult(
            status="not_found",
            message="None of the selected analysis files were found.",
            target_entity="AnalysisFiles",
            selected_ids=[],
            summary_df=summary_df,
            details=details,
        )

    if missing_ids:
        return DeletePreviewResult(
            status="partial",
            message=f"Preview generated for {len(found_ids)} analysis file(s). Missing ids: {missing_ids}",
            target_entity="AnalysisFiles",
            selected_ids=found_ids,
            summary_df=summary_df,
            details=details,
        )

    return DeletePreviewResult(
        status="ok",
        message=f"Preview generated for {len(found_ids)} analysis file(s).",
        target_entity="AnalysisFiles",
        selected_ids=found_ids,
        summary_df=summary_df,
        details=details,
    )


def execute_delete_analysis_files(
    conn: sqlite3.Connection,
    analysis_file_ids: Iterable[int | str],
) -> DeleteExecutionResult:
    preview = preview_delete_analysis_files(conn, analysis_file_ids)

    if preview.status in {"invalid_selection", "not_found"}:
        return DeleteExecutionResult(
            status=preview.status,
            message=preview.message,
            target_entity="AnalysisFiles",
            selected_ids=preview.selected_ids,
            deleted_counts={},
        )

    ids_to_delete = preview.selected_ids
    if not ids_to_delete:
        return DeleteExecutionResult(
            status="invalid_selection",
            message="No valid analysis file ids were available for deletion.",
            target_entity="AnalysisFiles",
            selected_ids=[],
            deleted_counts={},
        )

    try:
        with conn:
            deleted_counts = delete_queries.delete_analysis_files_cascade(conn, ids_to_delete)

        return DeleteExecutionResult(
            status="ok",
            message=(
                f"Deleted {deleted_counts.get('analysis_files_deleted', 0)} analysis file(s) and "
                f"{deleted_counts.get('results_deleted', 0)} orphaned result(s)."
            ),
            target_entity="AnalysisFiles",
            selected_ids=ids_to_delete,
            deleted_counts=deleted_counts,
        )
    except Exception as exc:
        return DeleteExecutionResult(
            status="error",
            message=f"Failed to delete analysis files: {exc}",
            target_entity="AnalysisFiles",
            selected_ids=ids_to_delete,
            deleted_counts={},
        )

# =========================================================
# Leaf / near-leaf delete previews
# =========================================================

def preview_delete_raw_files(
    conn: sqlite3.Connection,
    raw_file_ids: Iterable[int | str],
) -> DeletePreviewResult:
    try:
        normalized_ids = _validate_non_empty_ids(raw_file_ids, entity_name="raw file")
    except ValueError as exc:
        return DeletePreviewResult(
            status="invalid_selection",
            message=str(exc),
            target_entity="RawFiles",
            selected_ids=[],
            summary_df=_empty_df(["selected_count"]),
            details={},
        )

    raw_df = delete_queries.preview_delete_rows_by_ids(
        conn,
        table="RawFiles",
        ids=normalized_ids,
        id_column="id",
    )
    found_ids = _normalize_ids(raw_df["id"].tolist()) if not raw_df.empty and "id" in raw_df.columns else []
    missing_ids = sorted(set(normalized_ids) - set(found_ids))

    summary_df = delete_queries.preview_leaf_delete_summary(
        conn,
        table="RawFiles",
        ids=found_ids,
    )

    if not found_ids:
        return DeletePreviewResult(
            status="not_found",
            message="None of the selected raw files were found.",
            target_entity="RawFiles",
            selected_ids=[],
            summary_df=summary_df,
            details={"raw_files": raw_df},
        )

    if missing_ids:
        return DeletePreviewResult(
            status="partial",
            message=f"Preview generated for {len(found_ids)} raw file(s). Missing ids: {missing_ids}",
            target_entity="RawFiles",
            selected_ids=found_ids,
            summary_df=summary_df,
            details={"raw_files": raw_df},
        )

    return DeletePreviewResult(
        status="ok",
        message=f"Preview generated for {len(found_ids)} raw file(s).",
        target_entity="RawFiles",
        selected_ids=found_ids,
        summary_df=summary_df,
        details={"raw_files": raw_df},
    )


def preview_delete_tracking_files(
    conn: sqlite3.Connection,
    tracking_file_ids: Iterable[int | str],
) -> DeletePreviewResult:
    try:
        normalized_ids = _validate_non_empty_ids(tracking_file_ids, entity_name="tracking file")
    except ValueError as exc:
        return DeletePreviewResult(
            status="invalid_selection",
            message=str(exc),
            target_entity="TrackingFiles",
            selected_ids=[],
            summary_df=_empty_df(["selected_count"]),
            details={},
        )

    tracking_df = delete_queries.preview_delete_rows_by_ids(
        conn,
        table="TrackingFiles",
        ids=normalized_ids,
        id_column="id",
    )
    found_ids = _normalize_ids(tracking_df["id"].tolist()) if not tracking_df.empty and "id" in tracking_df.columns else []
    missing_ids = sorted(set(normalized_ids) - set(found_ids))

    summary_df = delete_queries.preview_leaf_delete_summary(
        conn,
        table="TrackingFiles",
        ids=found_ids,
    )

    if not found_ids:
        return DeletePreviewResult(
            status="not_found",
            message="None of the selected tracking files were found.",
            target_entity="TrackingFiles",
            selected_ids=[],
            summary_df=summary_df,
            details={"tracking_files": tracking_df},
        )

    if missing_ids:
        return DeletePreviewResult(
            status="partial",
            message=f"Preview generated for {len(found_ids)} tracking file(s). Missing ids: {missing_ids}",
            target_entity="TrackingFiles",
            selected_ids=found_ids,
            summary_df=summary_df,
            details={"tracking_files": tracking_df},
        )

    return DeletePreviewResult(
        status="ok",
        message=f"Preview generated for {len(found_ids)} tracking file(s).",
        target_entity="TrackingFiles",
        selected_ids=found_ids,
        summary_df=summary_df,
        details={"tracking_files": tracking_df},
    )


def preview_delete_masks(
    conn: sqlite3.Connection,
    mask_ids: Iterable[int | str],
) -> DeletePreviewResult:
    try:
        normalized_ids = _validate_non_empty_ids(mask_ids, entity_name="mask")
    except ValueError as exc:
        return DeletePreviewResult(
            status="invalid_selection",
            message=str(exc),
            target_entity="Masks",
            selected_ids=[],
            summary_df=_empty_df(["selected_count"]),
            details={},
        )

    mask_df = delete_queries.preview_delete_rows_by_ids(
        conn,
        table="Masks",
        ids=normalized_ids,
        id_column="id",
    )
    found_ids = _normalize_ids(mask_df["id"].tolist()) if not mask_df.empty and "id" in mask_df.columns else []
    missing_ids = sorted(set(normalized_ids) - set(found_ids))

    summary_df = delete_queries.preview_leaf_delete_summary(
        conn,
        table="Masks",
        ids=found_ids,
    )

    if not found_ids:
        return DeletePreviewResult(
            status="not_found",
            message="None of the selected masks were found.",
            target_entity="Masks",
            selected_ids=[],
            summary_df=summary_df,
            details={"masks": mask_df},
        )

    if missing_ids:
        return DeletePreviewResult(
            status="partial",
            message=f"Preview generated for {len(found_ids)} mask(s). Missing ids: {missing_ids}",
            target_entity="Masks",
            selected_ids=found_ids,
            summary_df=summary_df,
            details={"masks": mask_df},
        )

    return DeletePreviewResult(
        status="ok",
        message=f"Preview generated for {len(found_ids)} mask(s).",
        target_entity="Masks",
        selected_ids=found_ids,
        summary_df=summary_df,
        details={"masks": mask_df},
    )


def preview_delete_results(
    conn: sqlite3.Connection,
    result_ids: Iterable[int | str],
) -> DeletePreviewResult:
    try:
        normalized_ids = _validate_non_empty_ids(result_ids, entity_name="result")
    except ValueError as exc:
        return DeletePreviewResult(
            status="invalid_selection",
            message=str(exc),
            target_entity="Results",
            selected_ids=[],
            summary_df=_empty_df(["result_count", "result_analysis_link_count"]),
            details={},
        )

    details = delete_queries.preview_result_delete_details(conn, normalized_ids)
    summary_df = delete_queries.preview_result_delete_summary(conn, normalized_ids)

    results_df = details.get("results", pd.DataFrame())
    found_ids = _normalize_ids(results_df["id"].tolist()) if not results_df.empty and "id" in results_df.columns else []
    missing_ids = sorted(set(normalized_ids) - set(found_ids))

    if not found_ids:
        return DeletePreviewResult(
            status="not_found",
            message="None of the selected results were found.",
            target_entity="Results",
            selected_ids=[],
            summary_df=summary_df,
            details=details,
        )

    if missing_ids:
        return DeletePreviewResult(
            status="partial",
            message=f"Preview generated for {len(found_ids)} result(s). Missing ids: {missing_ids}",
            target_entity="Results",
            selected_ids=found_ids,
            summary_df=summary_df,
            details=details,
        )

    return DeletePreviewResult(
        status="ok",
        message=f"Preview generated for {len(found_ids)} result(s).",
        target_entity="Results",
        selected_ids=found_ids,
        summary_df=summary_df,
        details=details,
    )
# =========================================================
# Leaf deletes
# =========================================================

def execute_delete_raw_files(conn: sqlite3.Connection, raw_file_ids: Iterable[int | str]) -> DeleteExecutionResult:
    try:
        ids = _validate_non_empty_ids(raw_file_ids, entity_name="raw file")
    except ValueError as exc:
        return DeleteExecutionResult("invalid_selection", str(exc), "RawFiles", [], {})

    try:
        with conn:
            deleted = delete_queries.delete_raw_files_by_ids(conn, ids)
        return DeleteExecutionResult("ok", f"Deleted {deleted} raw file(s).", "RawFiles", ids, {"raw_files_deleted": deleted})
    except Exception as exc:
        return DeleteExecutionResult("error", f"Failed to delete raw files: {exc}", "RawFiles", ids, {})


def execute_delete_tracking_files(conn: sqlite3.Connection, tracking_file_ids: Iterable[int | str]) -> DeleteExecutionResult:
    try:
        ids = _validate_non_empty_ids(tracking_file_ids, entity_name="tracking file")
    except ValueError as exc:
        return DeleteExecutionResult("invalid_selection", str(exc), "TrackingFiles", [], {})

    try:
        with conn:
            deleted = delete_queries.delete_tracking_files_by_ids(conn, ids)
        return DeleteExecutionResult("ok", f"Deleted {deleted} tracking file(s).", "TrackingFiles", ids, {"tracking_files_deleted": deleted})
    except Exception as exc:
        return DeleteExecutionResult("error", f"Failed to delete tracking files: {exc}", "TrackingFiles", ids, {})


def execute_delete_masks(conn: sqlite3.Connection, mask_ids: Iterable[int | str]) -> DeleteExecutionResult:
    try:
        ids = _validate_non_empty_ids(mask_ids, entity_name="mask")
    except ValueError as exc:
        return DeleteExecutionResult("invalid_selection", str(exc), "Masks", [], {})

    try:
        with conn:
            deleted = delete_queries.delete_masks_by_ids(conn, ids)
        return DeleteExecutionResult("ok", f"Deleted {deleted} mask(s).", "Masks", ids, {"masks_deleted": deleted})
    except Exception as exc:
        return DeleteExecutionResult("error", f"Failed to delete masks: {exc}", "Masks", ids, {})


def execute_delete_results(conn: sqlite3.Connection, result_ids: Iterable[int | str]) -> DeleteExecutionResult:
    try:
        ids = _validate_non_empty_ids(result_ids, entity_name="result")
    except ValueError as exc:
        return DeleteExecutionResult("invalid_selection", str(exc), "Results", [], {})

    try:
        with conn:
            deleted_counts = delete_queries.delete_results_by_ids(conn, ids)
        return DeleteExecutionResult("ok", f"Deleted {deleted_counts.get('results_deleted', 0)} result(s).", "Results", ids, deleted_counts)
    except Exception as exc:
        return DeleteExecutionResult("error", f"Failed to delete results: {exc}", "Results", ids, {})


# =========================================================
# Restrictive delete for parent/reference tables
# =========================================================

def preview_delete_reference_entities(
    conn: sqlite3.Connection,
    *,
    parent_table: str,
    parent_ids: Iterable[int | str],
) -> DeletePreviewResult:
    if parent_table not in REFERENCE_PARENT_TABLES:
        return DeletePreviewResult(
            status="unsupported",
            message=f"Deletion preview is not supported for table '{parent_table}'.",
            target_entity=parent_table,
            selected_ids=[],
            summary_df=_empty_df([
                "selected_parent_count",
                "referencing_experiment_count",
                "deletable_parent_count",
                "blocked_parent_count",
            ]),
            details={},
        )

    try:
        ids = _validate_non_empty_ids(parent_ids, entity_name=parent_table)
    except ValueError as exc:
        return DeletePreviewResult(
            status="invalid_selection",
            message=str(exc),
            target_entity=parent_table,
            selected_ids=[],
            summary_df=_empty_df([
                "selected_parent_count",
                "referencing_experiment_count",
                "deletable_parent_count",
                "blocked_parent_count",
            ]),
            details={},
        )

    details = delete_queries.preview_reference_entity_delete(
        conn,
        parent_table=parent_table,
        parent_ids=ids,
    )
    summary_df = details["summary"]
    blocked = int(summary_df.iloc[0]["blocked_parent_count"]) if not summary_df.empty else 0

    if blocked > 0:
        msg = (
            f"Preview generated. {blocked} selected {parent_table} row(s) are still referenced by experiments "
            "and cannot be deleted."
        )
        status = "blocked"
    else:
        msg = f"Preview generated. Selected {parent_table} row(s) are deletable."
        status = "ok"

    return DeletePreviewResult(
        status=status,
        message=msg,
        target_entity=parent_table,
        selected_ids=ids,
        summary_df=summary_df,
        details=details,
    )


def execute_delete_reference_entities(
    conn: sqlite3.Connection,
    *,
    parent_table: str,
    parent_ids: Iterable[int | str],
) -> DeleteExecutionResult:
    preview = preview_delete_reference_entities(
        conn,
        parent_table=parent_table,
        parent_ids=parent_ids,
    )

    if preview.status in {"unsupported", "invalid_selection"}:
        return DeleteExecutionResult(
            status=preview.status,
            message=preview.message,
            target_entity=parent_table,
            selected_ids=preview.selected_ids,
            deleted_counts={},
        )

    try:
        with conn:
            deleted_counts = delete_queries.delete_reference_entities_restrict(
                conn,
                parent_table=parent_table,
                parent_ids=preview.selected_ids,
            )

        deleted = deleted_counts.get("deleted", 0)
        blocked = deleted_counts.get("blocked", 0)

        if deleted == 0 and blocked > 0:
            return DeleteExecutionResult(
                status="blocked",
                message=(
                    f"No {parent_table} rows were deleted because they are still referenced by experiments."
                ),
                target_entity=parent_table,
                selected_ids=preview.selected_ids,
                deleted_counts=deleted_counts,
            )

        return DeleteExecutionResult(
            status="ok",
            message=f"Deleted {deleted} {parent_table} row(s). Blocked {blocked}.",
            target_entity=parent_table,
            selected_ids=preview.selected_ids,
            deleted_counts=deleted_counts,
        )
    except Exception as exc:
        return DeleteExecutionResult(
            status="error",
            message=f"Failed to delete {parent_table}: {exc}",
            target_entity=parent_table,
            selected_ids=preview.selected_ids,
            deleted_counts={},
        )