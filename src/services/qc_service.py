
import sqlite3
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Optional

import pandas as pd

from queries import QC_queries


# =========================================================
# Standardized QC issue schema
# =========================================================

@dataclass(frozen=True)
class QCIssue:
    entity_type: str
    entity_id: str | int | None
    issue_category: str
    severity: str
    issue_summary: str
    issue_details: str


def issues_to_dataframe(issues: list[QCIssue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(
            columns=[
                "entity_type",
                "entity_id",
                "issue_category",
                "severity",
                "issue_summary",
                "issue_details",
            ]
        )
    return pd.DataFrame([asdict(i) for i in issues])


# =========================================================
# Small helpers
# =========================================================

def _safe_str(x) -> str:
    return "" if x is None else str(x)


def _experiment_label(row: pd.Series) -> str:
    parts = []
    if "date" in row and pd.notna(row["date"]):
        parts.append(f"date={row['date']}")
    if "replicate" in row and pd.notna(row["replicate"]):
        parts.append(f"replicate={row['replicate']}")
    if "organism" in row and pd.notna(row["organism"]):
        parts.append(f"organism={row['organism']}")
    if "protein" in row and pd.notna(row["protein"]):
        parts.append(f"protein={row['protein']}")
    if "condition" in row and pd.notna(row["condition"]):
        parts.append(f"condition={row['condition']}")
    if "capture_type" in row and pd.notna(row["capture_type"]):
        parts.append(f"capture_type={row['capture_type']}")
    return ", ".join(parts)


def _make_experiment_issue(
    row: pd.Series,
    *,
    category: str,
    severity: str,
    summary: str,
    details: str,
) -> QCIssue:
    entity_id = row.get("experiment_id", row.get("id"))
    return QCIssue(
        entity_type="Experiment",
        entity_id=entity_id,
        issue_category=category,
        severity=severity,
        issue_summary=summary,
        issue_details=details,
    )


# =========================================================
# Individual QC runners
# =========================================================

def qc_experiments_missing_files(
    conn: sqlite3.Connection,
    *,
    file_types: list[str] | tuple[str, ...] = ("raw", "tracking", "mask", "analysis"),
    filters: Optional[dict] = None,
    severity: str = "warning",
    limit: int = 500,
) -> pd.DataFrame:
    df = QC_queries.find_experiments_missing_files(
        conn=conn,
        file_types=file_types,
        filters=filters,
        limit=limit,
    )

    issues: list[QCIssue] = []
    file_types_txt = ", ".join(file_types)

    for _, row in df.iterrows():
        details = _experiment_label(row)
        issues.append(
            _make_experiment_issue(
                row,
                category="missing_files",
                severity=severity,
                summary=f"Experiment is missing one or more expected file types ({file_types_txt}).",
                details=details,
            )
        )

    return issues_to_dataframe(issues)


def qc_experiments_missing_metadata(
    conn: sqlite3.Connection,
    *,
    required_fields: list[str],
    filters: Optional[dict] = None,
    mode: str = "any",
    severity: str = "warning",
    limit: int = 500,
) -> pd.DataFrame:
    df = QC_queries.find_experiments_missing_metadata(
        conn=conn,
        required_fields=required_fields,
        filters=filters,
        mode=mode,
        limit=limit,
    )

    issues: list[QCIssue] = []
    fields_txt = ", ".join(required_fields)

    for _, row in df.iterrows():
        details = _experiment_label(row)
        issues.append(
            _make_experiment_issue(
                row,
                category="missing_metadata",
                severity=severity,
                summary=f"Experiment is missing required metadata fields ({fields_txt}).",
                details=details,
            )
        )

    return issues_to_dataframe(issues)


def qc_duplicate_experiments(
    conn: sqlite3.Connection,
    *,
    filters: Optional[dict] = None,
    severity: str = "warning",
) -> pd.DataFrame:
    df = QC_queries.find_duplicate_experiments(conn=conn, filters=filters)

    issues: list[QCIssue] = []
    for _, row in df.iterrows():
        issue_details = (
            f"organism={row.get('organism')}, "
            f"protein={row.get('protein')}, "
            f"strain={row.get('strain')}, "
            f"condition={row.get('condition')}, "
            f"capture_type={row.get('capture_type')}, "
            f"date={row.get('date')}, "
            f"replicate={row.get('replicate')}, "
            f"duplicate_count={row.get('duplicate_count')}, "
            f"experiment_ids={row.get('experiment_ids')}"
        )
        issues.append(
            QCIssue(
                entity_type="ExperimentGroup",
                entity_id=row.get("experiment_ids"),
                issue_category="duplicate_experiment",
                severity=severity,
                issue_summary="Multiple experiments share the same logical identity.",
                issue_details=issue_details,
            )
        )

    return issues_to_dataframe(issues)


def qc_missing_values(
    conn: sqlite3.Connection,
    *,
    main_table: str,
    requested_columns: list[str],
    missing_columns: list[str] | str,
    filters: Optional[dict] = None,
    mode: str = "any",
    severity: str = "warning",
    limit: int = 500,
) -> pd.DataFrame:
    df = QC_queries.find_missing_values(
        conn=conn,
        main_table=main_table,
        requested_columns=requested_columns,
        missing_columns=missing_columns,
        filters=filters,
        mode=mode,
        limit=limit,
    )

    issues: list[QCIssue] = []
    missing_txt = ", ".join(missing_columns if isinstance(missing_columns, list) else [missing_columns])

    id_col = None
    for candidate in [f"{main_table.lower()}_id", "id", "experiment_id", "user_id"]:
        if candidate in df.columns:
            id_col = candidate
            break

    for _, row in df.iterrows():
        entity_id = row.get(id_col) if id_col else None
        details = ", ".join([f"{col}={row[col]}" for col in df.columns if col != id_col])
        issues.append(
            QCIssue(
                entity_type=main_table,
                entity_id=entity_id,
                issue_category="missing_values",
                severity=severity,
                issue_summary=f"Row has missing values in one or more target fields ({missing_txt}).",
                issue_details=details,
            )
        )

    return issues_to_dataframe(issues)


def qc_analysis_files_without_results(
    conn: sqlite3.Connection,
    *,
    severity: str = "warning",
    limit: int = 500,
) -> pd.DataFrame:
    df = QC_queries.find_analysis_files_without_results(conn=conn, limit=limit)

    issues: list[QCIssue] = []
    for _, row in df.iterrows():
        details = (
            f"file_name={row.get('file_name')}, "
            f"file_type={row.get('file_type')}, "
            f"file_path={row.get('file_path')}"
        )
        issues.append(
            QCIssue(
                entity_type="AnalysisFile",
                entity_id=row.get("analysis_file_id"),
                issue_category="missing_results_link",
                severity=severity,
                issue_summary="Analysis file is not linked to any result.",
                issue_details=details,
            )
        )

    return issues_to_dataframe(issues)


def qc_results_without_analysis_files(
    conn: sqlite3.Connection,
    *,
    severity: str = "warning",
    limit: int = 500,
) -> pd.DataFrame:
    df = QC_queries.find_results_without_analysis_files(conn=conn, limit=limit)

    issues: list[QCIssue] = []
    for _, row in df.iterrows():
        details = (
            f"result_type={row.get('result_type')}, "
            f"result_value={row.get('result_value')}, "
            f"sample_size={row.get('sample_size')}, "
            f"standard_error={row.get('standard_error')}"
        )
        issues.append(
            QCIssue(
                entity_type="Result",
                entity_id=row.get("result_id"),
                issue_category="missing_analysis_file_link",
                severity=severity,
                issue_summary="Result is not linked to any analysis file.",
                issue_details=details,
            )
        )

    return issues_to_dataframe(issues)


def qc_experiments_with_analysis_but_no_results(
    conn: sqlite3.Connection,
    *,
    filters: Optional[dict] = None,
    severity: str = "warning",
    limit: int = 500,
) -> pd.DataFrame:
    df = QC_queries.find_experiments_with_analysis_but_no_results(
        conn=conn,
        filters=filters,
        limit=limit,
    )

    issues: list[QCIssue] = []
    for _, row in df.iterrows():
        details = ", ".join(f"{col}={row[col]}" for col in df.columns if col != "experiment_id")
        issues.append(
            QCIssue(
                entity_type="Experiment",
                entity_id=row.get("experiment_id"),
                issue_category="analysis_without_results",
                severity=severity,
                issue_summary="Experiment has analysis files but no linked results.",
                issue_details=details,
            )
        )

    return issues_to_dataframe(issues)


def qc_incomplete_linked_entities(
    conn: sqlite3.Connection,
    *,
    base_table: str,
    present_entity: tuple[str, str],
    missing_entity: tuple[str, str],
    present_bridge: Optional[tuple[str, str, str]] = None,
    missing_bridge: Optional[tuple[str, str, str]] = None,
    filters: Optional[dict] = None,
    severity: str = "warning",
    limit: int = 500,
    summary: Optional[str] = None,
) -> pd.DataFrame:
    df = QC_queries.find_incomplete_linked_entities(
        conn=conn,
        base_table=base_table,
        present_entity=present_entity,
        missing_entity=missing_entity,
        present_bridge=present_bridge,
        missing_bridge=missing_bridge,
        filters=filters,
        limit=limit,
    )

    issues: list[QCIssue] = []
    issue_summary = summary or (
        f"{base_table} has {present_entity[0]} but is missing {missing_entity[0]}."
    )

    id_col = f"{base_table.lower()}_id"

    for _, row in df.iterrows():
        details = ", ".join(f"{col}={row[col]}" for col in df.columns if col != id_col)
        issues.append(
            QCIssue(
                entity_type=base_table,
                entity_id=row.get(id_col),
                issue_category="incomplete_links",
                severity=severity,
                issue_summary=issue_summary,
                issue_details=details,
            )
        )

    return issues_to_dataframe(issues)


# =========================================================
# Combined runner
# =========================================================

def run_default_qc_suite(
    conn: sqlite3.Connection,
    *,
    experiment_filters: Optional[dict] = None,
    limit_per_check: int = 500,
) -> pd.DataFrame:
    """
    Run a default QC suite and return one standardized issue table.
    """
    outputs = [
        qc_experiments_missing_files(
            conn,
            file_types=("raw", "tracking", "mask", "analysis"),
            filters=experiment_filters,
            limit=limit_per_check,
        ),
        qc_experiments_missing_metadata(
            conn,
            required_fields=["organism", "protein", "condition", "capture_type", "date", "replicate"],
            filters=experiment_filters,
            mode="any",
            limit=limit_per_check,
        ),
        qc_duplicate_experiments(
            conn,
            filters=experiment_filters,
        ),
        qc_experiments_with_analysis_but_no_results(
            conn,
            filters=experiment_filters,
            limit=limit_per_check,
        ),
        qc_analysis_files_without_results(
            conn,
            limit=limit_per_check,
        ),
        qc_results_without_analysis_files(
            conn,
            limit=limit_per_check,
        ),
    ]

    non_empty = [df for df in outputs if not df.empty]
    if not non_empty:
        return issues_to_dataframe([])

    out = pd.concat(non_empty, ignore_index=True)
    return out