import json
import sqlite3
import logging
from typing import Any, Dict, Optional, Tuple, List
import csv
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)



# -----------------------
# Data Normalization utilities
# -----------------------


def norm_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).lower().strip()
    return s if s else None

def norm_float(x: Any) -> Optional[float]:
    if x in ("", None):
        return None
    try:
        v = float(x)
        return None if v == 0.0 else round(v, 6)
    except Exception:
        return None

def norm_int(x: Any) -> Optional[int]:
    if x in ("", None):
        return None
    try:
        v = int(x)
        return None if v == 0 else v
    except Exception:
        return None

def strip_text(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None

def json_text(x: Any) -> Optional[str]:
    if x in ("", None):
        return None
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    # if it's already a string
    s = str(x).strip()
    return s if s else None

# -----------------------
# DB helpers
# -----------------------
def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def get_or_create_id(
    cur: sqlite3.Cursor,
    table: str,
    unique_fields: Dict[str, Any],
) -> int:
    """
    unique_fields must match the UNIQUE constraint columns, or at least be stable.
    NULL handling:
      - For nullable unique fields: SQLite UNIQUE treats NULL as distinct,
        so we avoid passing NULLs into unique_fields to achieve true "same row".
    """
    cols = list(unique_fields.keys())
    vals = [unique_fields[c] for c in cols]

    # alternative way: 
    # where = " AND ".join([f"{c} IS ?" if v is None else f"{c} = ?" for c, v in zip(cols, vals)])
    # cur.execute(f"SELECT id FROM {table} WHERE {where}", vals)

    where_parts = []
    values = []

    for c, v in zip(cols, vals):
        if v is None:
            where_parts.append(f"{c} IS NULL")
        else:
            where_parts.append(f"{c} = ?")
            values.append(v)

    where = " AND ".join(where_parts)

    cur.execute(
        f"SELECT id FROM {table} WHERE {where}",
        tuple(values)
    )
    row = cur.fetchone()
    if row:
        return int(row[0])
    placeholders = ", ".join(["?"] * len(cols))
    col_sql = ", ".join(cols)
    cur.execute(f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})", vals)
    logger.info(f"Inserted new value in the {table} table  with values: {unique_fields}")
    return int(cur.lastrowid)

def _build_where(unique_fields: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Builds WHERE clause that correctly handles NULL comparisons.
    """
    parts = []
    values = []
    for k, v in unique_fields.items():
        if v is None:
            parts.append(f"{k} IS NULL")
        else:
            parts.append(f"{k} = ?")
            values.append(v)
    return " AND ".join(parts), values


def insert_or_skip(
    cur: sqlite3.Cursor,
    *,
    table: str,
    unique_fields: Dict[str, Any],
    insert_fields: Dict[str, Any],
    context: Dict[str, Any],
    reason_prefix: str,
) -> Dict[str, Any]:
    """
    Returns a dict:
      - status: 'inserted' | 'skipped'
      - reason (if skipped)
      - existing_id / new_id
      - table
      - unique_fields
      - context (file info etc.)
    """
    where_sql, where_vals = _build_where(unique_fields)
    cur.execute(f"SELECT id FROM {table} WHERE {where_sql}", where_vals)
    row = cur.fetchone()

    if row:
        return {
            "status": "skipped",
            "reason": f"{reason_prefix}: duplicate",
            "existing_id": int(row[0]),
            "table": table,
            "unique_fields": unique_fields,
            "context": context,
        }

    cols = ", ".join(insert_fields.keys())
    placeholders = ", ".join(["?"] * len(insert_fields))
    cur.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
        list(insert_fields.values()),
    )
    return {
        "status": "inserted",
        "new_id": int(cur.lastrowid),
        "table": table,
        "unique_fields": unique_fields,
        "context": context,
    }
    
# -----------------------
# Main: manifest insertion
# -----------------------
def insert_manifest(manifest: Dict[str, Any], db_path: str) -> Dict[str, Any]:
    """
    Inserts a single manifest into the DB.
    Returns: experiment_id
    """
    exp = manifest.get("experiment") or {}
    g = manifest.get("global_defaults") or {}
    files = manifest.get("files_resolved") or []

    if not exp:
        logger.error("Manifest missing 'experiment'.")
        raise ValueError("Manifest missing 'experiment'.")
        
    if not files:
        logger.error("Manifest has no 'files_resolved' rows.")
        raise ValueError("Manifest has no 'files_resolved' rows.")

    conn = get_conn(db_path)
    cur = conn.cursor()
    try:
        conn.execute("BEGIN;")

        # ---- lookups for already existing records in different tables ----
        organism_id = get_or_create_id(cur, "Organism", {"organism_name": norm_text(exp.get("organism"))})
        protein_id  = get_or_create_id(cur, "Protein", {"protein_name": norm_text(exp.get("protein"))})
        strain_id   = get_or_create_id(cur, "StrainOrCellLine", {"strain_name": norm_text(exp.get("strain"))})

        condition_id = get_or_create_id(cur, "Condition", {
            "condition_name": norm_text(exp.get("condition_name")),
            "concentration_value": norm_float(exp.get("concentration_value")),
            "concentration_unit": norm_text(exp.get("concentration_unit")),
        })

        user_id = get_or_create_id(cur, "User", {
            "name": norm_text(g.get("user_name")),
            "last_name": norm_text(g.get("user_last_name")),
            "email": norm_text(g.get("user_email")),
        })

        # CaptureSetting: built from resolved experiment+global overrides
        capture_setting_id = get_or_create_id(cur, "CaptureSetting", {
            "capture_type": norm_text(exp.get("capture_type")),
            "exposure_time": norm_float(exp.get("exposure_time")),
            "time_interval": norm_float(exp.get("time_interval")),

            "fluorescent_dye": norm_text(exp.get("fluorescent_dye") or g.get("fluorescent_dye")),
            "dye_concentration_value": norm_float(exp.get("dye_concentration_value") or g.get("dye_concentration_value")),
            "dye_concentration_unit": norm_text(exp.get("dye_concentration_unit") or g.get("dye_concentration_unit")),
            "laser_wavelength": norm_float(exp.get("laser_wavelength") or g.get("laser_wavelength")),
            "laser_intensity": norm_float(exp.get("laser_intensity") or g.get("laser_intensity")),
            "camera_binning": norm_int(exp.get("camera_binning") or g.get("camera_binning")),
            "objective_magnification": norm_float(exp.get("objective_magnification") or g.get("objective_magnification")),
            "pixel_size": norm_float(exp.get("pixel_size") or g.get("pixel_size")),
        })

        # ---- experiment ----
        experiment_lookup = {
            "organism_id": organism_id,
            "protein_id": protein_id,
            "strain_id": strain_id,
            "condition_id": condition_id,
            "capture_setting_id": capture_setting_id,
            "user_id": user_id,
            "date": norm_text(exp.get("date")),
            "replicate": norm_int(exp.get("replicate")),
        }

        #where = " AND ".join([f"{k} IS ?" if v is None else f"{k} = ?" for k, v in experiment_lookup.items()])
        #cur.execute(f"SELECT id FROM Experiment WHERE {where}", list(experiment_lookup.values()))
        # second options is to build where with correct NULL handling:
        where_sql, where_vals = _build_where(experiment_lookup)
        cur.execute(f"SELECT id FROM Experiment WHERE {where_sql}", where_vals)
        row = cur.fetchone()

        if row:
            experiment_id = int(row[0])
        else:
            insert_fields = dict(experiment_lookup)
            insert_fields.update({
                "is_valid": int(bool(exp.get("is_valid", True))),
                "comment": norm_text(exp.get("comment")),
                "experiment_path": norm_text(exp.get("experiment_path")),
            })
            cols = ", ".join(insert_fields.keys())
            placeholders = ", ".join(["?"] * len(insert_fields))
            cur.execute(f"INSERT INTO Experiment ({cols}) VALUES ({placeholders})", list(insert_fields.values()))
            experiment_id = int(cur.lastrowid)

        report = {
            "experiment_id": experiment_id,
            "inserted_counts": {"raw": 0, "tracking": 0, "mask": 0, "analysis": 0},
            "skipped": [],
        }

        # ---- files ----
        for f in files:
            dt = f.get("data_type")
            if dt == "raw":
                outcome = _insert_raw(cur, experiment_id, f)
            elif dt == "tracking":
                outcome = _insert_tracking(cur, experiment_id, f)
            elif dt == "mask":
                outcome = _insert_mask(cur, experiment_id, f)
            elif dt in ("analysis", "batch_analysis", "plot", "config"):
                outcome = _insert_analysis(cur, experiment_id, f)
            else:
                # ignore/unassigned should not appear here; we filtered ignore already
                continue
            bucket = "analysis" if dt in ("analysis","batch_analysis","plot","config") else dt
            if outcome["status"] == "inserted":
                report["inserted_counts"][bucket] += 1
                logger.info(f"Inserted {bucket} file: {outcome['context']}")

            else:
                report["skipped"].append(outcome)
                logger.info(f"Skipped {bucket} file due to duplicate: {outcome['context']}")

        conn.commit()
        logger.info("data was succussfully inserted to the db.")
        return report
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    

# -----------------------
# Table-specific inserts
# -----------------------
def _insert_raw(cur: sqlite3.Cursor, experiment_id: int, f: Dict[str, Any]) -> Dict[str, Any]:

    unique_fields = {
        "experiment_id": experiment_id,
        "file_name": norm_text(f.get("file_name")),
        "file_path": norm_text(f.get("path")),  # handle NULL as discussed
    }
    insert_fields = {
        **unique_fields,
        "file_type": norm_text(f.get("ext")),
    }
    return insert_or_skip(
        cur,
        table="RawFiles",
        unique_fields=unique_fields,
        insert_fields=insert_fields,
        context={"data_type": "raw", "file_name": f.get("file_name"), "path": f.get("path")},
        reason_prefix="RawFiles",
    )


def _insert_tracking(cur: sqlite3.Cursor, experiment_id: int, f: Dict[str, Any]) -> Dict[str, Any]:
    trackmate_params = f.get("trackmate_settings_json") or {}
    unique_fields = {
        "experiment_id": experiment_id,
        "file_name": norm_text(f.get("file_name")),
        "threshold": f.get("threshold"),
        "linking_distance": f.get("linking_distance"),
        "gap_closing_distance": f.get("gap_closing_distance"),
        "max_frame_gap": f.get("max_frame_gap"),
    }
    insert_fields = {
        **unique_fields,
        "file_type": norm_text(f.get("ext")),
        "file_path": f.get("path"),
        "trackmate_settings_json": norm_text(json.dumps(trackmate_params)) if isinstance(trackmate_params, dict) else trackmate_params,
    }
    return insert_or_skip(
        cur,
        table="TrackingFiles",
        unique_fields=unique_fields,
        insert_fields=insert_fields,
        context={"data_type": "tracking", "file_name": f.get("file_name"), "path": f.get("path")},
        reason_prefix="TrackingFiles",
    )

def _insert_mask(cur: sqlite3.Cursor, experiment_id: int, f: Dict[str, Any]) -> Dict[str, Any]:
    seg_params = f.get("segmentation_parameters") or {}
    unique_fields = {
        "experiment_id": experiment_id,
        "mask_name": norm_text(f.get("file_name")),
        "segmentation_method": norm_text(f.get("segmentation_method")),
    }
    insert_fields = {
        **unique_fields,
        "mask_type": norm_text(f.get("mask_type")),
        "mask_path": norm_text(f.get("path")),
        "segmentation_parameters": norm_text(json.dumps(seg_params)) if isinstance(seg_params, dict) else seg_params,
    }

    return insert_or_skip(
        cur,
        table="Masks",
        unique_fields=unique_fields,
        insert_fields=insert_fields,
        context={"data_type": "mask", "file_name": f.get("file_name"), "path": f.get("path")},
        reason_prefix="Masks",
    )

def _insert_analysis(cur: sqlite3.Cursor, experiment_id: int, f: Dict[str, Any]) -> Dict[str, Any]:
    # depends on whether AnalysisFiles is global + link table, or has experiment_id directly
    # Here I assume you have a link table like before.

    unique_fields = {
        "file_name": norm_text(f.get("file_name")),
        "file_path": norm_text(f.get("path")),
    }
    
    insert_fields = {
        **unique_fields,
        "file_type": norm_text(f.get("ext")),
    }

    
    
    result = insert_or_skip(
        cur,
        table="AnalysisFiles",
        unique_fields=unique_fields,
        insert_fields=insert_fields,
        context={"data_type": "analysis", "file_name": f.get("file_name"), "path": f.get("path")},
        reason_prefix="AnalysisFiles",
    )

    

    if result["status"] == "inserted":
        # Insert into link table
        analysis_file_id = cur.execute("SELECT last_insert_rowid()").fetchone()[0]
        cur.execute(
            "INSERT INTO ExperimentAnalysisFiles (experiment_id, analysis_file_id) VALUES (?, ?)",
            (experiment_id, analysis_file_id)
        )
        result["linked_analysis_file_id"] = analysis_file_id

    else:
        # If it already exists, we need to ensure the link exists (idempotent)
        analysis_file_id = result["existing_id"]
        cur.execute(
            "SELECT 1 FROM ExperimentAnalysisFiles WHERE experiment_id = ? AND analysis_file_id = ?",
            (experiment_id, analysis_file_id)
        )
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO ExperimentAnalysisFiles (experiment_id, analysis_file_id) VALUES (?, ?)",
                (experiment_id, analysis_file_id)
            )
        result["linked_analysis_file_id"] = analysis_file_id
        logger.info(f"Linking existing AnalysisFile id {analysis_file_id} to experiment {experiment_id}")
        
    return result
    