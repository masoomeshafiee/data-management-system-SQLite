"""
Centralized module for updating records in the database.
"""

from __future__ import annotations

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, Mapping
import logging
from queries import COLUMN_MAP, build_query_context, find_invalid_categorical_values, TABLE_RELATIONSHIPS
import os
import re
import time
import math

IDENTITY_MAP = { # not used (optional)
    # Core biological entities
    "Organism": ["organism_name"],
    "Protein": ["protein_name"],
    "StrainOrCellLine": ["strain_name"],

    # Experimental metadata
    "Condition": ["condition_name", "concentration_value", "concentration_unit"],
    "User": ["user_name"],

    # Capture settings — uniquely defined by imaging parameters
    "CaptureSetting": [
        "capture_type", 
        "exposure_time", 
        "time_interval", 
        "dye_concentration_value"
    ],

    # Experimental runs — one unique combination per replicate / capture setup

    # Raw and processed data files
    "RawFiles": ["file_name", "file_type", "experiment_id"],
    "TrackingFiles": ["file_name", "file_type", "experiment_id"],
    "Masks": ["file_name", "file_type", "mask_type", "experiment_id"],
    "AnalysisFiles": ["file_name", "file_type", "experiment_id"],
    "AnalysisResults": ["result_type", "analysis_file_id"],

    # Linking / associative tables
    "ExperimentAnalysisFiles": ["experiment_id", "analysis_file_id"],
    "AnalysisResultExperiments": ["experiment_id", "analysis_result_id"],
}

MAIN_MINIMAL_KEYS = { # not used (optional)
    "Experiment": ["date", "replicate"],

    # Join tables: identity = just their FKs (no extra minimal keys)
    "ExperimentAnalysisFiles": [],
    "AnalysisResultExperiments": [],

    # File-like tables (need local natural keys *and* experiment_id FK)
    "RawFiles": ["file_name", "file_type"],
    "TrackingFiles": ["file_name", "file_type"],
    "Masks": ["file_name", "file_type", "mask_type"],

    # If you use these as main tables:
    "AnalysisFiles": ["file_name", "file_type"],
    "AnalysisResults": ["result_type"],
}
# CSV/update-dict aliases for identity columns (left = target table, right = mapping: identity_col -> alt key in update_dict)
IDENTITY_ALIASES = {
    "Condition": {
        "condition_name": "condition",
        "concentration_value": "concentration_value",
        "concentration_unit": "concentration_unit",
    },
    "CaptureSetting": {
        "capture_type": "capture_type",
        "exposure_time": "exposure_time",
        "time_interval": "time_interval",
        "dye_concentration_value": "dye_concentration_value",
    },
    "User": {
        "user_name": "user_name",
    },
    "Organism": {"organism_name": "organism"},
    "Protein": {"protein_name": "protein"},
    "StrainOrCellLine": {"strain_name": "strain"},
}

# Which extra columns are safe to update in existing FK rows
MUTABLE_EXTRAS = {
    "CaptureSetting": [
        "dye_concentration_unit",
        "fluorescent_dye",
        "laser_wavelength",
        "laser_intensity",
        "camera_binning",
        "objective_magnification",
        "pixel_size"
    ],
    "Condition": [
        "concentration_unit",
    ],
    # Add others if ever needed...
}

# ----------------------------------------------------------------------
#                     Categorical values
# ----------------------------------------------------------------------
ALLOWED_CATEGORICALS: Dict[str, Set[str]] = {"is_valid": {"Y", "N"}, "organism_name": {"E.coli", "yeast", "Human cells"}, "capture_type":{"fast", "long", "confocal"},
"dye_concentration_unit":{"uM", "nM", "pM", "n/a"}, # number with % sign, e.g. "20%"
"condition_name":{"cpt", "uv", "hu", "untreated"},
"concentration_unit":{"mM", "uM", "nM", "pM", "j/m2", "n/a"}
}  # etc.

# ----------------------- Logging setup -----------------------
logger = logging.getLogger("update_records")
if not logger.handlers:
    fh = logging.FileHandler("../data/db_update.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)


# ----------------------- Helper -----------------------

# --------------------------------------------------------------------------------------
# Connection helper (enforce good pragmas)
# --------------------------------------------------------------------------------------

def connect_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

# --------------------------------------------------------------------------------------
# Schema + FK discovery
# --------------------------------------------------------------------------------------
def get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cur.fetchall()]  # name at index 1

def get_fk_info(conn: sqlite3.Connection, main_table: str) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping: fk_col -> {"table": target_table, "to_col": referenced_col}
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA foreign_key_list({main_table});")
    return {row[3]: {"table": row[2], "to_col": row[4]} for row in cur.fetchall()}

def get_multi_table_schema(conn: sqlite3.Connection, main_table: str, fk_map: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    tables = [main_table] + [fk["table"] for fk in fk_map.values()]
    out = {}
    for t in tables:
        out[t] = get_table_columns(conn, t)
    return out

# --------------------------------------------------------------------------------------
# CSV / dataframe validation
# --------------------------------------------------------------------------------------
def validate_columns_strict(df: pd.DataFrame, table_columns: Dict[str, List[str]]) -> None:
    """
    Strict: CSV columns must exist in *some* table (main or related).
    """
    all_cols: Set[str] = set().union(*table_columns.values()) if table_columns else set()
    invalid = [c for c in df.columns if c not in all_cols]
    if invalid:
        raise ValueError(f"Invalid columns in CSV (not found in schema): {invalid}")

# --------------------------------------------------------------------------------------
# FK resolution (safe, partial-aware)
# --------------------------------------------------------------------------------------

def sqlite_supports_returning(conn: sqlite3.Connection) -> bool:
    # SQLite supports RETURNING since 3.35 (2021). Expose a quick check.
    v = conn.execute("select sqlite_version()").fetchone()[0]
    major, minor, patch = (int(x) for x in v.split("."))
    return (major, minor, patch) >= (3, 35, 0)

def resolve_or_create_fk(
    conn: sqlite3.Connection,
    target_table: str,
    row: pd.Series,
    table_columns: Dict[str, List[str]],
    identity_map: Dict[str, List[str]],
    on_missing_identity: str = "skip",      # "skip" | "error"
    allow_extra_attrs: bool = True,         # include extras on INSERT
    existing_policy: str = "keep",          # "keep" | "update" | "error"
    mutable_extras: Dict[str, List[str]] | None = None,  # whitelisted cols per table when updating
    dry_run=True,                # if True, don't actually modify DB
    dry_run_id_seed=-1000001    # optional - synthetic ID for dry-run inserts must be negative value
):
    """
    - If identity exists:
        keep:   return id, ignore extras
        update: UPDATE allowed extra columns (whitelist) if provided
        error:  raise if provided extras differ from stored values
    - If identity missing:
        INSERT identity (+ extras if allow_extra_attrs)
    Returns:
        - int (real id) when FK exists or was created
        - None when identity is incomplete and on_missing_identity != "error"
        - (synthetic_id, payload) when dry_run=True and FK would be created
    """
    if target_table not in identity_map:
        raise ValueError(f"No identity defined for table '{target_table}'.")

    ident_cols = identity_map[target_table]
    missing_needed = [c for c in ident_cols if c not in row.index or pd.isna(row[c])]
    if missing_needed:
        if on_missing_identity == "error":
            raise ValueError(f"{target_table}: missing required identity fields {missing_needed}")
        return None

    identity_values = {c: row[c] for c in ident_cols}
    cur = conn.cursor()
    where_id = " AND ".join([f"{c} = ?" for c in ident_cols])
    cur.execute(f"SELECT * FROM {target_table} WHERE {where_id}", tuple(identity_values[c] for c in ident_cols))
    existing = cur.fetchone()
    colnames = [d[0] for d in cur.description] if existing else table_columns.get(target_table, [])

    # Build payload = identity (+ extras if provided & valid)
    payload = dict(identity_values)
    if allow_extra_attrs:
        valid_cols = set(table_columns.get(target_table, []))
        for c in row.index:
            if c in valid_cols and c not in payload and pd.notna(row[c]):
                payload[c] = row[c]
    # ------------------------------------------------------------------
    # CASE 1: identity already exists
    # ----------------------------------------------------------------
    if existing:
        # Existing row found
        row_dict = dict(zip(colnames, existing))
        if existing_policy == "keep":
            return int(row_dict["id"])
        elif existing_policy == "error":
            # if any provided extra differs from stored, raise
            diffs = {}
            for k, v in payload.items():
                if k in ident_cols:  # skip identity—by definition equal
                    continue
                if k in row_dict and row_dict[k] != v:
                    diffs[k] = (row_dict[k], v)
            if diffs:
                raise ValueError(
                    f"{target_table} identity exists, but provided extra fields differ: {diffs}. "
                    f"Either change identity, or set existing_policy='update', or omit extras."
                )
            return int(row_dict["id"])
        elif existing_policy == "update":
            # Only allow whitelisted columns to be updated
            allowed = set((mutable_extras or {}).get(target_table, []))
            to_update = {
                k: v for k, v in payload.items()
                if k not in ident_cols and k in allowed and k in row_dict and row_dict[k] != v
            }
            if to_update:
                if dry_run:
                    logger.info(
                        f"[DRY-RUN][FK UPDATE] {target_table} id={row_dict['id']} would update: {list(to_update.keys())}"
                    )
                    # don't actually update
                else:
                    # named placeholders for both SET and WHERE
                    set_clause = ", ".join(f"{k} = :set_{k}" for k in to_update)
                    where_named = " AND ".join(f"{c} = :id_{c}" for c in ident_cols)

                    params = {}
                    params.update({f"set_{k}": v for k, v in to_update.items()})
                    params.update({f"id_{c}": identity_values[c] for c in ident_cols})

                    sql = f"UPDATE {target_table} SET {set_clause} WHERE {where_named}"
                    conn.execute(sql, params)  # <-- dict for named params

                    logger.info(f"[FK UPDATE] {target_table} id={row_dict['id']} updated: {list(to_update.keys())}")
            return int(row_dict["id"])

        else:
            raise ValueError("existing_policy must be 'keep', 'update', or 'error'.")

    # ------------------------------------------------------------------
    # CASE 2: identity does NOT exist -> would INSERT
    # ------------------------------------------------------------------
    # No existing row → INSERT with identity (+ extras)
    if dry_run:
        # just say "we would create one", return a synthetic ID
        # you can also log the payload for the preview
        logger.info(f"[DRY-RUN][FK INSERT] Would insert into {target_table}: {payload}")
        return int(dry_run_id_seed), payload
    cols = list(payload.keys())
    vals = [payload[c] for c in cols]
    placeholders = ", ".join(["?"] * len(vals))

    # Use your UNIQUE(identity) and RETURNING when supported
    on_conflict_cols = f"({', '.join(ident_cols)})"
    do_update_set = ", ".join([f"{c}={c}" for c in ident_cols]) or "id=id"

    if sqlite_supports_returning(conn):
        sql = (
            f"INSERT INTO {target_table} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT{on_conflict_cols} DO UPDATE SET {do_update_set} RETURNING id"
        )
        new_id = conn.execute(sql, vals).fetchone()[0]
        return int(new_id)
    else:
        try:
            conn.execute(f"INSERT INTO {target_table} ({', '.join(cols)}) VALUES ({placeholders})", vals)
            return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        except sqlite3.IntegrityError:
            cur.execute(f"SELECT id FROM {target_table} WHERE {where_id}", tuple(identity_values[c] for c in ident_cols))
            r2 = cur.fetchone()
            if r2:
                return int(r2[0])
            raise

# -----------------------------------------------------------------------
# Relational preview with JOINs
# -----------------------------------------------------------------------
def get_relational_record_preview(
    conn: sqlite3.Connection,
    main_table: str,
    key_column: str,
    record_id: Any,
    df_row: pd.Series,
    fk_map: Dict[str, Dict[str, str]],
    table_columns: Dict[str, List[str]],
) -> pd.DataFrame:
    touched = {main_table}
    for fk_col, fk in fk_map.items():
        tgt = fk["table"]
        if any(col in table_columns.get(tgt, []) for col in df_row.index):
            touched.add(tgt)

    # Do NOT use filters={'id': ...}; your filter parser doesn't know 'id'.
    where_clauses, params, joins = build_query_context(
        main_table=main_table,
        filters=None,
        extra_where=[f"{main_table}.{key_column} = :pk"],
        extra_params={"pk": record_id},
        base_tables={main_table},
        required_tables_extra=touched - {main_table},
    )

    query = f"SELECT * FROM {main_table} " + " ".join(joins)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    return pd.read_sql_query(query, conn, params=params)

# --------------------------------------------------------------------------------------
# Main update runner (dry-run supported)
# --------------------------------------------------------------------------------------
#-----------------------------------Helpers--------------------------------------------
def _values_different(a, b) -> bool:
    """Robust equality: NaN==NaN treated equal; numeric, strings vs numbers compared numerically; else ==."""
    # both NaN -> equal
    if pd.isna(a) and pd.isna(b):
        return False
    # one NaN only -> different
    if pd.isna(a) ^ pd.isna(b):
        return True
    # try numeric compare
    try:
        return float(a) != float(b)
    except Exception:
        pass
    # fallback direct compare
    return a != b
# -----------------------------------------------------------------------------------------------
def _fetch_main_row_dict(conn: sqlite3.Connection, main_table: str, key_column: str, rec_id):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({main_table});")
    cols = [r[1] for r in cur.fetchall()]
    cur.execute(f"SELECT {', '.join(cols)} FROM {main_table} WHERE {key_column} = ?", (rec_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip(cols, row))
# -------------------------------------------Main function ----------------------------------------------------
def run_updates_from_dataframe(
    conn: sqlite3.Connection,
    main_table: str,
    df_updates: pd.DataFrame,
    key_column: str = "id",
    dry_run: bool = True,
    output_dir: str = "../data/update_previews",
) -> Tuple[Path, int, int]:
    """
    Returns: (out_dir, n_previewed, n_missing)
    """
    if key_column not in df_updates.columns:
        raise ValueError(f"CSV must include '{key_column}' column.")

    # Discover FK and schema
    fk_map = get_fk_info(conn, main_table)                           # {fk_col: {"table": target, "to_col": "id"}}
    schema = get_multi_table_schema(conn, main_table, fk_map)        # {table: [cols...]}

    # Strict validation that all CSV columns exist in *some* known table
    validate_columns_strict(df_updates, schema)

    # Prepare output folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) / f"updates_{main_table}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    updated_rows: List[pd.DataFrame] = []
    missing_rows: List[pd.Series] = []

    cur = conn.cursor()

    # Wrap in a transaction (even for dry-run we won't commit)
    try:
        pending_fk_payloads = {}
        for _, row in df_updates.iterrows():
            rec_id = row[key_column]

            # 1) existence check
            cur.execute(f"SELECT 1 FROM {main_table} WHERE {key_column} = ?", (rec_id,))
            if not cur.fetchone():
                missing_rows.append(row)
                continue


            # Fetch current main-table row once for robust diffing
            current_main = _fetch_main_row_dict(conn, main_table, key_column, rec_id)
            if current_main is None:
                # defensive, though we already checked existence
                missing_rows.append(row)
                continue

            # 2) Resolve FKs
            #    Prefer explicit *_id from CSV; otherwise, attempt identity-based resolution
            updates_main: Dict[str, Any] = {}
            for fk_col, fk in fk_map.items():
                target = fk["table"]

               # (A) CSV provides FK id directly → trust it, but only include if different
                if fk_col in row.index and pd.notna(row[fk_col]):
                    try:
                        new_fk = int(row[fk_col])
                    except (ValueError, TypeError):
                        raise ValueError(f"{fk_col} must be an integer id; got {row[fk_col]!r}")
                    old_fk = current_main.get(fk_col)
                    if _values_different(old_fk, new_fk):
                        updates_main[fk_col] = new_fk
                    continue
                # (B) No id given — attempt identity-based resolution ONLY if the row
                #     actually contains any columns for the target table.
                has_any_target_fields = any(
                    (c in row.index and pd.notna(row[c]))
                    for c in schema.get(target, [])
                )
                if not has_any_target_fields:
                    # Row says nothing about this related table → don’t touch this FK.
                    continue

                # Use strict resolver (requires full identity to insert; dedup via UNIQUE+UPSERT)
                fk_result = resolve_or_create_fk(
                    conn=conn,
                    target_table=target,
                    row=row,
                    table_columns=schema,
                    identity_map=IDENTITY_MAP,      # make sure this is defined/imported
                    on_missing_identity="skip",      # skip if identity incomplete (don’t error)
                    allow_extra_attrs=True,
                    dry_run=dry_run,
                    dry_run_id_seed=-1
                )
                if isinstance(fk_result, tuple):
                    fk_id, fk_payload = fk_result
                    updates_main[fk_col] = fk_id
                    pending_fk_payloads[(target, fk_id)] = fk_payload

                else:
                    fk_id = fk_result
                    if fk_id is not None:
                        old_fk = current_main.get(fk_col)
                        if _values_different(old_fk, fk_id):
                            updates_main[fk_col] = fk_id
                    else:
                        logger.warning(
                            f"[FK] Skipping resolution for {fk_col} on {main_table}.{key_column}={rec_id} "
                            f"(no id provided and identity incomplete for {target})."
                        )

            # 3) Direct main-table updates (skip PK + FK columns) / only add if changed
            main_cols = set(schema[main_table])
            forbidden = {key_column, *fk_map.keys()}
            for col in (c for c in row.index if c in main_cols and c not in forbidden):
                if pd.notna(row[col]):
                    new_val = row[col]
                    old_val = current_main.get(col)
                    if _values_different(old_val, new_val):
                        updates_main[col] = new_val

            # 4) Relational preview (JOINs)
            df_old = get_relational_record_preview(
                conn=conn,
                main_table=main_table,
                key_column=key_column,
                record_id=rec_id,
                df_row=row,
                fk_map=fk_map,
                table_columns=schema,
            )

            if updates_main:
                # annotate only the actually-changed fields
                if df_old.empty:
                    df_old = pd.DataFrame([{key_column: rec_id}])
                for col, new_val in updates_main.items():
                    df_old[f"new_{col}"] = new_val
                    if col in df_old.columns:
                        df_old[f"change_{col}"] = df_old[col].astype(str) + " -> " + str(new_val)
                updated_rows.append(df_old)

            # 5) Apply update (if not dry-run)
            if not dry_run and updates_main:
                set_clause = ", ".join(f"{c} = :{c}" for c in updates_main)
                params = {**updates_main, key_column: rec_id}
                cur.execute(
                    f"UPDATE {main_table} SET {set_clause} WHERE {key_column} = :{key_column}",
                    params,
                )

        if not dry_run:
            conn.commit()

    except Exception:
        if not dry_run:
            conn.rollback()
        raise

    # 6) Write previews
    n_previewed = 0
    if updated_rows:
        # build the combined preview first
        preview_df = pd.concat(updated_rows, ignore_index=True)
        # inject readable values for synthetic (dry-run) FK inserts
        if pending_fk_payloads and updated_rows:
            rels = get_fk_info(conn, main_table)  # {fk_col: {"table": target, ...}}
            preview_df = pd.concat(updated_rows, ignore_index=True)

            for idx, row in preview_df.iterrows():
                for fk_col, fk in rels.items():
                    target_table = fk["table"]
                    if fk_col not in preview_df.columns:
                        continue
                    fk_val = row[fk_col]
                    if pd.isna(fk_val):
                        continue
                    # safe cast
                    try:
                        fk_int = int(fk_val)
                    except (ValueError, TypeError):
                        continue
                    # only handle synthetic (negative) IDs
                    if fk_int >= 0:
                        continue

                    key = (target_table, fk_int)
                    payload = pending_fk_payloads.get(key)
                    if not payload:
                        continue

                    # figure out base name (condition_id -> condition)
                    base = fk_col[:-3] if fk_col.endswith("_id") else fk_col
                    ident_cols = IDENTITY_MAP.get(target_table, [])
                    for c in ident_cols:
                        new_col = f"new_{base}_{c}"
                        if new_col not in preview_df.columns:
                            preview_df[new_col] = None
                        preview_df.at[idx, new_col] = payload.get(c)

                    status_col = f"{base}_fk_status"
                    if status_col not in preview_df.columns:
                        preview_df[status_col] = None
                    preview_df.at[idx, status_col] = "would create (dry-run)"
        preview_path = out_dir / f"{main_table}_update_preview.csv"
        preview_df.to_csv(preview_path, index=False)
        logger.info(f"[Preview] wrote: {preview_path}")
        n_previewed = len(updated_rows)

    n_missing = 0
    if missing_rows:
        missing_df = pd.DataFrame(missing_rows)
        missing_path = out_dir / "missing_records.csv"
        missing_df.to_csv(missing_path, index=False)
        logger.info(f"[Missing] wrote: {missing_path}")
        n_missing = len(missing_rows)

    if dry_run:
        logger.info("Dry-run: no DB changes applied.")

    return out_dir, n_previewed, n_missing

# --------------------------------------------------------------------------------------
# Convenience runner for CSV path
# --------------------------------------------------------------------------------------
def run_updates_from_csv(
    db_path: str,
    main_table: str,
    csv_path: str,
    key_column: str = "id",
    *,
    identity_cols: List[str] = [],
    on_ambiguous: str = "error",
    create_missing_main: bool = False,
    commit_missing_fk: bool = True,
    dry_run: bool = True,
    output_dir: str = "../data/update_previews",
):
    df = pd.read_csv(csv_path)
    with connect_sqlite(db_path) as conn:
        if key_column in df.columns:
            # Normal path: CSV already has IDs
            return run_updates_from_dataframe(
                conn=conn,
                main_table=main_table,
                df_updates=df,
                key_column=key_column,
                dry_run=dry_run,
                output_dir=output_dir,
            )
        # Delegate to the no-id path
        if not key_column:
            logger.error("No key_column specified.")
            raise ValueError("key_column must be specified when CSV lacks IDs.")

# ----------------------------------------------------------------
# Safe execution using an update dictionary with preview
# --------------------------------------------------------------

def update_records_from_csv_relational_strict(
    db_path,
    main_table,
    csv_path,
    key_column="id",
    commit_missing_fk=True,
    dry_run=True,
    output_dir="../data/update_previews"
):
    """
    Strict relational CSV-based database updater with full JOIN preview.

    Features:
      - Enforces exact column names (no guessing)
      - Validates CSV columns against DB schema
      - Resolves or inserts missing FK entities
      - Logs partial FK inserts when not all fields are provided
      - Provides relational preview with JOINs across linked tables
      - Dry-run mode (safe, no writes)
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    df_updates = pd.read_csv(csv_path)
    if key_column not in df_updates.columns:
        raise ValueError(f"CSV must include '{key_column}' column.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir) / f"{Path(csv_path).stem}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    updated_rows, missing_records = [], []

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        # --- Get FK info for the main table ---
        cur.execute(f"PRAGMA foreign_key_list({main_table});")
        fk_info = {fk[3]: {"table": fk[2], "to_col": fk[4]} for fk in cur.fetchall()}

        # --- Load schema for all related tables ---
        table_columns = {}
        for tbl in [main_table] + [fk["table"] for fk in fk_info.values()]:
            cur.execute(f"PRAGMA table_info({tbl});")
            table_columns[tbl] = [col[1] for col in cur.fetchall()]

        # --- Validate CSV columns ---
        all_valid_columns = set().union(*table_columns.values())
        invalid_cols = [c for c in df_updates.columns if c not in all_valid_columns]
        if invalid_cols:
            raise ValueError(f"Invalid columns in CSV: {invalid_cols}")

        # --- Iterate through CSV rows ---
        for _, row in df_updates.iterrows():
            record_id = row[key_column]

            # Check if record exists
            cur.execute(f"SELECT COUNT(*) FROM {main_table} WHERE {key_column} = ?", (record_id,))
            if cur.fetchone()[0] == 0:
                missing_records.append(row)
                continue

            updates_main = {}

            # --- Step 1: Resolve FK relationships ---
            for fk_col, fk in fk_info.items():
                target_table = fk["table"]
                related_cols = [c for c in df_updates.columns if c in table_columns[target_table]]
                if not related_cols:
                    continue

                nested_values = {
                    col: row[col] for col in related_cols if not pd.isna(row[col])
                }
                if not nested_values:
                    continue

                # Try to find existing FK record
                where_clause = " AND ".join([f"{k} = ?" for k in nested_values])
                cur.execute(f"SELECT id FROM {target_table} WHERE {where_clause}", tuple(nested_values.values()))
                result = cur.fetchone()

                if result:
                    fk_id = result[0]
                elif commit_missing_fk:
                    cols, vals = zip(*nested_values.items())
                    placeholders = ", ".join(["?"] * len(vals))
                    cur.execute(f"INSERT INTO {target_table} ({', '.join(cols)}) VALUES ({placeholders})", vals)
                    fk_id = cur.lastrowid

                    # Warn if partial insert
                    missing_fields = set(table_columns[target_table]) - set(nested_values.keys())
                    if missing_fields:
                        logger.warning(f"Partial insert into {target_table}: missing fields {missing_fields}")
                    logger.info(f"Inserted new {target_table}: {nested_values} (id={fk_id})")
                else:
                    fk_id = None

                if fk_id:
                    updates_main[fk_col] = fk_id

            # --- Step 2: Direct main table updates ---
            for col in table_columns[main_table]:
                if col in (key_column, *fk_info.keys()):
                    continue
                if col in df_updates.columns and not pd.isna(row[col]):
                    updates_main[col] = row[col]

            # --- Step 3: Build relational preview using join context ---
            touched_tables = {main_table}
            for fk_col, fk in fk_info.items():
                target_table = fk["table"]
                related_cols = [c for c in df_updates.columns if c in table_columns.get(target_table, [])]
                if related_cols:
                    touched_tables.add(target_table)

            where_clauses, params, joins = build_query_context(
                main_table=main_table,
                filters={key_column: record_id},
                base_tables={main_table},
                required_tables_extra=touched_tables - {main_table}
            )

            query = f"SELECT * FROM {main_table} " + " ".join(joins)
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            df_old = pd.read_sql_query(query, conn, params=params)

            # --- Step 4: Generate preview diff ---
            if not df_old.empty:
                for col, new_val in updates_main.items():
                    df_old[f"new_{col}"] = new_val
                    if col in df_old.columns:
                        df_old[f"change_{col}"] = df_old[col].astype(str) + " → " + str(new_val)
                updated_rows.append(df_old)

            # --- Step 5: Apply updates ---
            if not dry_run and updates_main:
                set_clause = ", ".join([f"{col} = :{col}" for col in updates_main])
                params = updates_main | {key_column: record_id}
                cur.execute(f"UPDATE {main_table} SET {set_clause} WHERE {key_column} = :{key_column}", params)

        if not dry_run:
            conn.commit()

    # --- Save outputs ---
    if updated_rows:
        pd.concat(updated_rows, ignore_index=True).to_csv(out_dir / f"{main_table}_update_preview.csv", index=False)
        print(f"✅ {len(updated_rows)} updates previewed. Saved to {out_dir}")
    if missing_records:
        pd.DataFrame(missing_records).to_csv(out_dir / f"missing_records.csv", index=False)
        print(f"⚠️ {len(missing_records)} missing IDs logged.")
    if dry_run:
        print("ℹ️ Dry-run mode: no DB changes applied.")

# ----------------------- Update functions -----------------------
# -------------------------Helpers-------------------------

def _fetch_main_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    return [r[1] for r in cur.fetchall()]

def _validate_update_cols(conn: sqlite3.Connection, main_table: str, update_dict: dict):
    main_cols = set(_fetch_main_cols(conn, main_table))
    bad = [c for c in update_dict if c not in main_cols]
    if bad:
        raise ValueError(f"Columns not in {main_table}: {bad}")
    
# ----------------------------Update records by filter -----------------------------
#--------------------------------------------------------------
#                                   Main update execution
#--------------------------------------------------------------
def expand_fk_columns_in_preview(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    table_relationships: dict[str, dict[str, str]],
    identity_map: dict[str, list[str]],
    main_table: str,
    key_column: str = "id",
) -> pd.DataFrame:
    """
    Replace *_id columns in a preview DataFrame with readable identity info from the related tables.
    For example, replace condition_id=3 with condition="HU", concentration_value=11, etc.
    """
    if df.empty:
        return df

    rels = table_relationships.get(main_table, {})
    out_df = df.copy()

    for fk_col, target_table in rels.items():
        if fk_col not in out_df.columns:
            continue
        ids = out_df[fk_col].dropna().unique().tolist()
        if not ids:
            continue

        ident_cols = identity_map.get(target_table, [])
        if not ident_cols:
            continue  # nothing to expand

        id_placeholders = ", ".join(["?"] * len(ids))
        cols_sql = ", ".join(["id"] + ident_cols)
        q = f"SELECT {cols_sql} FROM {target_table} WHERE id IN ({id_placeholders})"
        fk_df = pd.read_sql_query(q, conn, params=ids)

        # Prefix each identity column with the FK base name (e.g., condition_name, condition_concentration_value)
        base = fk_col.replace("_id", "")
        fk_df = fk_df.rename(
            columns={c: f"{base}_{c}" for c in ident_cols if c != "id"}
        )

        out_df = out_df.merge(fk_df, how="left", left_on=fk_col, right_on="id")
        out_df.drop(columns=["id"], inplace=True, errors="ignore")

    return out_df

def execute_update(
    db_path: str,
    *,
    main_table: str,
    key_column: str,
    update_dict: dict,
    # choose one of the following to select targets:
    id_subquery_sql: str | None = None,   # e.g. "SELECT Experiment.id FROM Experiment JOIN ... WHERE ..."
    id_subquery_params: dict | tuple | None = None,
    id_list: list[int] | None = None,
    # preview/joins
    preview_output_dir: str = "../data/update_previews",
    dry_run: bool = True,
    chunk_size: int = 500,                # for large IN (...) lists
    pending_fk_payloads: dict | None = None,
):
    """
    Generic UPDATE engine:
      - Validates update columns
      - Resolves target IDs via subquery or explicit list
      - Computes diffs (only updates rows that actually change)
      - Emits a JOIN-aware preview CSV (new_/change_ only for changed columns)
      - Applies the UPDATE (unless dry_run)
    Returns: (out_dir, n_changed, n_targeted)
    """
    import pandas as pd
    if not update_dict:
        raise ValueError("update_dict is empty; nothing to update")
    
    conn = sqlite3.connect(db_path)

    _validate_update_cols(conn, main_table, update_dict)

    cur = conn.cursor()

    # 1) Resolve target IDs
    if id_list is None:
        if not id_subquery_sql:
            raise ValueError("Provide id_list or id_subquery_sql")
        import pandas as pd
        id_df = pd.read_sql_query(id_subquery_sql, conn, params=id_subquery_params or {})
        if id_df.empty:
            # no targets
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path(preview_output_dir) / f"updates_{main_table}_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"No target IDs found from subquery {id_subquery_sql}.")
            return out_dir, 0, 0
        id_list = id_df.iloc[:, 0].dropna().astype(int).tolist()
    else:
        id_list = [int(x) for x in id_list if pd.notna(x)]

    if not id_list:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(preview_output_dir) / f"updates_{main_table}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("No target IDs provided in the ID list.")
        return out_dir, 0, 0

    n_targeted = len(id_list)

    # 2) Load current values only for columns we care about
    cols_needed = [key_column] + list(update_dict.keys())
    cols_sql = ", ".join(cols_needed)
    # use chunking to avoid very large IN lists
    current_rows = []
    for i in range(0, len(id_list), chunk_size):
        chunk = id_list[i:i+chunk_size]
        placeholders = ", ".join(["?"] * len(chunk))
        q = f"SELECT {cols_sql} FROM {main_table} WHERE {key_column} IN ({placeholders})"
        current_rows.append(pd.read_sql_query(q, conn, params=chunk))
    current_df = pd.concat(current_rows, ignore_index=True) if current_rows else pd.DataFrame(columns=cols_needed)

    if current_df.empty:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(preview_output_dir) / f"updates_{main_table}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir, 0, n_targeted

    # 3) Compute diffs -> keep only truly changing IDs
    def row_changes(s: pd.Series) -> dict:
        changes = {}
        for col, new_val in update_dict.items():
            old_val = s.get(col, None)
            if _values_different(old_val, new_val):
                changes[col] = new_val
        return changes

    current_df["_changes"] = current_df.apply(row_changes, axis=1)
    current_df["_has_change"] = current_df["_changes"].apply(bool)
    changed_ids = current_df.loc[current_df["_has_change"], key_column].astype(int).tolist()

    # 4) Prepare preview (JOIN-aware) only for changed IDs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(preview_output_dir) / f"updates_{main_table}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_changed = len(changed_ids)
    if n_changed == 0:
        # nothing to update
        preview_path = out_dir / f"{main_table}_filter_update_preview.csv"
        pd.DataFrame({key_column: id_list}).to_csv(preview_path, index=False)
        return out_dir, 0, n_targeted

    # --- Build the joined preview with a stable PK alias ---
    named_ids = {f"id{i}": v for i, v in enumerate(changed_ids)}
    id_placeholders = ", ".join([f":id{i}" for i in range(len(changed_ids))])

    where, params, joins = build_query_context(
        main_table=main_table,
        filters=None,
        extra_where=[f"{main_table}.{key_column} IN ({id_placeholders})"],
        extra_params=named_ids,                 # IMPORTANT: dict, not tuple
        base_tables={main_table},
    )

    # Select main PK twice: as __pk__ (stable) and the full main table columns
    select_cols = f"{main_table}.{key_column} AS __pk__, {main_table}.*"
    q = f"SELECT {select_cols} FROM {main_table} " + " ".join(joins)
    if where:
        q += " WHERE " + " AND ".join(where)

    df_old = pd.read_sql_query(q, conn, params=params)

    # Optionally expand FKs to readable identity columns
    df_old = expand_fk_columns_in_preview(
        conn=conn,
        df=df_old,
        table_relationships=TABLE_RELATIONSHIPS,
        identity_map=IDENTITY_MAP,
        main_table=main_table,
        key_column=key_column
    )

    # Choose a robust index: prefer main 'id' if present, else '__pk__'
    pk_for_index = key_column if key_column in df_old.columns else "__pk__"
    df_old_indexed = df_old.set_index(pk_for_index)

    # quick lookup of changes per id (this stays as-is)
    changes_map = {
        int(s[key_column]): s["_changes"]
        for _, s in current_df[current_df["_has_change"]].iterrows()
    }

    # annotate only the actually changed columns
    for col in update_dict.keys():
        # add only if any row actually changes this column
        if any(col in ch for ch in changes_map.values()):
            new_vals = []
            change_text = []
            for rid in df_old_indexed.index:
                ch = changes_map.get(int(rid), {})
                if col in ch:
                    nv = ch[col]
                    new_vals.append(nv)
                    old = df_old_indexed.loc[rid, col] if col in df_old_indexed.columns else None
                    change_text.append(f"{old} -> {nv}")
                else:
                    new_vals.append(None)
                    change_text.append(None)
            df_old[f"new_{col}"] = new_vals
            if col in df_old.columns:
                df_old[f"change_{col}"] = change_text
    rels = TABLE_RELATIONSHIPS.get(main_table, {})
    for fk_col, target_table in rels.items():
        # only proceed if this FK column is part of this update
        if fk_col not in update_dict:
            continue

        # find rows where THIS fk actually changed
        changed_rows_for_fk = [rid for rid, ch in changes_map.items() if fk_col in ch]
        if not changed_rows_for_fk:
            continue

        # fetch identity columns for the target table
        ident_cols = IDENTITY_MAP.get(target_table, [])
        if not ident_cols:
            continue

        # pull the set of *new* FK ids we need to resolve
        
        new_ids = [changes_map[rid][fk_col] for rid in changed_rows_for_fk]
        unique_new_ids = sorted(set(int(x) for x in new_ids if pd.notna(x)))
        if not unique_new_ids:
            continue

        # load identity values for the *new* FK ids
        placeholders = ", ".join(["?"] * len(unique_new_ids))
        cols_sql = ", ".join(["id"] + ident_cols)
        fk_new_df = pd.read_sql_query(
            f"SELECT {cols_sql} FROM {target_table} WHERE id IN ({placeholders})",
            conn,
            params=unique_new_ids,
        )
        # map: new_fk_id -> {ident_col: value}
        new_identity_map = {
            int(r["id"]): {c: r[c] for c in ident_cols} for _, r in fk_new_df.iterrows()
        }

        base = fk_col[:-3] if fk_col.endswith("_id") else fk_col  # e.g., "condition"

        # add new_<base>_<identcol> and change_<base>_<identcol> for each identity column
        for c in ident_cols:
            new_col = f"new_{base}_{c}"
            change_col = f"change_{base}_{c}"
            new_vals = []
            change_vals = []

            for rid in df_old_indexed.index:
                ch = changes_map.get(int(rid), {})
                if fk_col in ch:
                    new_fk_id = int(ch[fk_col])
                    new_val = new_identity_map.get(new_fk_id, {}).get(c)

                    # old readable value is already in df_old thanks to expand_fk_columns_in_preview
                    old_readable_col = f"{base}_{c}"
                    old_val = (
                        df_old_indexed.loc[rid, old_readable_col]
                        if old_readable_col in df_old_indexed.columns
                        else None
                    )
                    new_vals.append(new_val)
                    change_vals.append(f"{old_val} -> {new_val}")
                else:
                    new_vals.append(None)
                    change_vals.append(None)

            df_old[new_col] = new_vals
            df_old[change_col] = change_vals

    # ensure preview shows an 'id' column even if it was shadowed by joins
    if "id" not in df_old.columns and "__pk__" in df_old.columns:
        df_old.rename(columns={"__pk__": "id"}, inplace=True)

    preview_path = out_dir / f"{main_table}_filter_update_preview.csv"
    if pending_fk_payloads:
        rels = TABLE_RELATIONSHIPS.get(main_table, {})
        for idx, row in df_old.iterrows():
            for fk_col, target_table in rels.items():
                if fk_col not in df_old.columns:
                    continue
                fk_val = row[fk_col]
                if pd.isna(fk_val):
                    continue
                try:
                    fk_int = int(fk_val)
                except (ValueError, TypeError):
                    continue
                if fk_int >= 0:
                    continue  # real FK

                key = (target_table, fk_int)
                payload = pending_fk_payloads.get(key)
                if not payload:
                    continue

                base = fk_col[:-3] if fk_col.endswith("_id") else fk_col
                ident_cols = IDENTITY_MAP.get(target_table, [])
                for c in ident_cols:
                    new_col = f"new_{base}_{c}"
                    if new_col not in df_old.columns:
                        df_old[new_col] = None
                    df_old.at[idx, new_col] = payload.get(c)

                status_col = f"{base}_fk_status"
                if status_col not in df_old.columns:
                    df_old[status_col] = None
                df_old.at[idx, status_col] = "would create (dry-run)"

    df_old.to_csv(preview_path, index=False)


    # 5) Apply the update (only changed ids)
    if not dry_run:
        # We’ll update in chunks if needed
        set_clause = ", ".join([f"{c} = :{c}" for c in update_dict])
        for i in range(0, len(changed_ids), chunk_size):
            chunk = changed_ids[i:i+chunk_size]
            placeholders = ", ".join([f":id{i}_{j}" for j in range(len(chunk))])
            sql = (
                f"UPDATE {main_table} SET {set_clause} "
                f"WHERE {key_column} IN ({placeholders})"
            )
            # build params namespace
            params = dict(update_dict)
            params.update({f"id{i}_{j}": v for j, v in enumerate(chunk)})
            conn.execute(sql, params)
        conn.commit()

    return out_dir, n_changed, n_targeted
    

# ----------------------------------------------------------------
# 1. Update records by filter
# ----------------------------------------------------------------

def preprocess_update_dict_for_fk(
    conn: sqlite3.Connection,
    main_table: str,
    update_dict: dict,
    *,
    table_relationships: dict[str, dict[str, str]],
    identity_map: dict[str, list[str]],
    table_columns: dict[str, list[str]],
    on_missing_identity: str = "error",     # "error" | "skip"
    allow_extra_attrs: bool = True,
    identity_aliases: dict[str, dict[str, str]] | None = None,
    dry_run: bool = True,
    dry_run_id_seed: int = -1,
    pending_fk_payloads: dict | None = None
) -> tuple[dict, dict]:
    """
    Convert natural-key specs present in update_dict into main-table *_id fields.

    Behavior:
      - If *_id is present in update_dict -> trust it and remove any provided target-table fields.
      - If any identity field for a target table is provided -> require all identity fields,
        build a row-like payload with identity + ANY provided target-table columns (extras),
        resolve/create the FK row, set *_id, and strip ALL target-table fields from update_dict.
      - Extras are applied ONLY on insert (resolve_or_create_fk handles that). Existing rows are not mutated.
    """
    new_ud = dict(update_dict)
    rels = table_relationships.get(main_table, {})

    for fk_col, target_table in rels.items():
        # 0) If caller already supplied the FK id, keep it and strip any target-table fields
        if fk_col in new_ud and new_ud[fk_col] is not None:
            # remove any provided columns that actually belong to the target table
            for c in list(new_ud.keys()):
                if c in table_columns.get(target_table, []):
                    new_ud.pop(c, None)
            # also remove alias keys for identity, if any
            if identity_aliases and target_table in identity_aliases:
                for alias in identity_aliases[target_table].values():
                    new_ud.pop(alias, None)
            continue

        ident_cols = identity_map.get(target_table, [])
        if not ident_cols:
            continue  # we don't know how to resolve this table by identity

        # 1) Gather ALL provided fields for the target table (identity + extras)
        provided_target_fields: dict[str, object] = {}
        for c in table_columns.get(target_table, []):
            if c in new_ud and new_ud[c] is not None:
                provided_target_fields[c] = new_ud[c]

        # also pull identity via aliases if present (for identity only)
        if identity_aliases and target_table in identity_aliases:
            for ident_col, alias_key in identity_aliases[target_table].items():
                if ident_col not in provided_target_fields and alias_key in new_ud and new_ud[alias_key] is not None:
                    provided_target_fields[ident_col] = new_ud[alias_key]

        # If nothing in this update targets the FK table, skip
        any_identity_present = any(k in provided_target_fields for k in ident_cols)
        if not any_identity_present:
            continue

        # 2) Ensure full identity present if any identity was provided
        missing = [c for c in ident_cols if c not in provided_target_fields or provided_target_fields[c] is None]
        if missing:
            if on_missing_identity == "error":
                raise ValueError(
                    f"Update references '{target_table}' but identity is incomplete. "
                    f"Provided: {sorted([k for k in provided_target_fields.keys() if k in ident_cols])}; "
                    f"missing: {missing}. Required identity: {ident_cols}."
                )
            # on 'skip', ignore this FK entirely (leave naturals; execute_update will later reject them)
            continue

        # 3) Resolve/create FK row using identity + extras (resolver will include extras only on insert)
        row_like = pd.Series(provided_target_fields)
        pending_fk_payloads = pending_fk_payloads or {}
        fk_result = resolve_or_create_fk(
            conn=conn,
            target_table=target_table,
            row=row_like,
            table_columns=table_columns,
            identity_map=identity_map,
            on_missing_identity="error",
            existing_policy="update", # Note: change this if you want different behavior on existing rows in the fk table 
            mutable_extras=MUTABLE_EXTRAS,
            allow_extra_attrs=allow_extra_attrs,  # extras are accepted on insert path
            dry_run=dry_run,
            dry_run_id_seed=-1
        )
        if isinstance(fk_result, tuple):
            fk_id, payload = fk_result
            pending_fk_payloads[(target_table, fk_id)] = payload
        else:
            fk_id = fk_result
            
            if fk_id is None:
                # shouldn't happen with on_missing_identity="error"
                continue
        new_ud[fk_col] = int(fk_id)

        # 5) Strip ALL target-table fields (identity + extras) from the update dict
        for c in list(new_ud.keys()):
            if c in table_columns.get(target_table, []):
                new_ud.pop(c, None)
        # and strip alias keys if they were used
        if identity_aliases and target_table in identity_aliases:
            for alias in identity_aliases[target_table].values():
                new_ud.pop(alias, None)

    return new_ud, pending_fk_payloads


def update_records_by_filter(db_path, main_table, update_dict, filters=None, key_column = "id", limit=None, dry_run=True, output_dir="../data/update_previews"):
    """
    General update: set fields in update_dict where filters match.
    Example: update_records_by_filter(DB_PATH, "Experiment",
             {"is_valid": "Y"}, {"user_name": "Masoumeh"})
    """
    filters = filters or {}
    with sqlite3.connect(db_path) as conn:
        # Build subquery selecting target ids
        where, params, joins = build_query_context(
            main_table=main_table,
            filters=filters,
            base_tables={main_table},
        )
        sub = f"SELECT {main_table}.{key_column} FROM {main_table} " + " ".join(joins)
        if where:
            sub += " WHERE " + " AND ".join(where)
        if limit:
            sub += f" LIMIT {limit}"

        # --- NEW: preprocess update_dict to repoint FKs based on natural keys
        fk_map = get_fk_info(conn, main_table)
        schema = get_multi_table_schema(conn, main_table, fk_map)

        logical_rels = TABLE_RELATIONSHIPS.get(main_table, {})
        for _, target_table in logical_rels.items():
            if target_table not in schema:
                schema[target_table] = get_table_columns(conn, target_table)

        new_ud, pending_fk_payloads = preprocess_update_dict_for_fk(
            conn=conn,
            main_table=main_table,
            update_dict=update_dict,
            table_relationships=TABLE_RELATIONSHIPS,
            identity_map=IDENTITY_MAP,
            table_columns=schema,
            on_missing_identity="error",       # require full identity if any identity field present
            allow_extra_attrs=True,
            identity_aliases=IDENTITY_ALIASES,
            dry_run=dry_run,
            dry_run_id_seed=-1,
            pending_fk_payloads={}
        )

        # Now execute_update sees only main-table columns (e.g., condition_id), so it won't error
        return execute_update(
            db_path,
            main_table=main_table,
            key_column=key_column,
            update_dict=new_ud,
            id_subquery_sql=sub,
            id_subquery_params=params,
            preview_output_dir=output_dir,
            dry_run=dry_run,
            pending_fk_payloads=pending_fk_payloads,
        )
# -------------------------------------------------------------------------
# 3. Update missing values with defaults
# -------------------------------------------------------------------------
def update_missing_values_with_default(
    db_path: str,
    main_table: str,
    column_defaults: dict[str, object],   # {"comment": "(none)", "is_valid": 1}
    *,
    filters: dict | None = None,
    key_column: str = "id",
    dry_run: bool = True,
    output_dir: str = "../data/update_previews",
    missing_predicates: dict[str, str] | None = None,   # optional per-column SQL predicate
):
    """
    For each (column -> default_value), update rows where the column is "missing".
    Default "missing" predicate:
      - TEXT-like:  (col IS NULL OR TRIM(col) = '')
      - otherwise:  (col IS NULL)

    You can override per column via missing_predicates, e.g.:
      {"time_interval": "time_interval IS NULL OR time_interval <= 0"}
    """
    filters = filters or {}
    missing_predicates = missing_predicates or {}
    # One run id so all per-column folders are grouped under a single parent
    run_id = time.strftime("%Y%m%d-%H%M%S")
    parent_out_dir = os.path.join(output_dir, f"{main_table}__defaults__{run_id}")
    os.makedirs(parent_out_dir, exist_ok=True)


    
    total_changed = 0
    total_targeted = 0
    per_column_dirs: dict[str, str] = {}

    with sqlite3.connect(db_path) as conn:
        # Validate columns against main table schema
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({main_table});")
        main_cols = {r[1]: r for r in cur.fetchall()}

        for col, default_val in column_defaults.items():
            if col not in main_cols:
                raise ValueError(f"Column '{col}' not in {main_table}")

            # Missing predicate
            if col in missing_predicates:
                miss_sql = missing_predicates[col]
            else:
                # Heuristic: TEXT affinity → treat empty string as missing
                type_aff = (main_cols[col][2] or "").upper()
                is_texty = any(x in type_aff for x in ("CHAR", "CLOB", "TEXT"))
                if is_texty:
                    miss_sql = f"({main_table}.{col} IS NULL OR LENGTH(TRIM({main_table}.{col})) = 0)"
                else:
                    miss_sql = f"({main_table}.{col} IS NULL OR {main_table}.{col} = '')"

            # Build ID subquery with your filter model + missing condition
            where, params, joins = build_query_context(
                main_table=main_table,
                filters=filters,
                base_tables={main_table},
                extra_where=[miss_sql],
            )
            sub = f"SELECT {main_table}.{key_column} FROM {main_table} " + " ".join(joins)
            if where:
                sub += " WHERE " + " AND ".join(where)

            # UNIQUE preview folder per column → no overwrites
            col_out_dir = os.path.join(parent_out_dir, f"{col}")
            os.makedirs(col_out_dir, exist_ok=True)

            # Run an isolated execute_update per column (keeps preview precise)
            update_dict = {col: default_val}
            out_dir_this, n_changed, n_targeted = execute_update(
                db_path,
                main_table=main_table,
                key_column=key_column,
                update_dict=update_dict,
                id_subquery_sql=sub,
                id_subquery_params=params,
                preview_output_dir=col_out_dir,
                dry_run=dry_run,
            )
            per_column_dirs[col] = out_dir_this or col_out_dir
            total_changed += n_changed
            total_targeted += n_targeted

    return parent_out_dir, per_column_dirs, total_changed, total_targeted

# --------------------------------------------------------------------------
#                           Fix invalid foreign keys
# ----------------------------------------------------------
def update_invalid_foreign_keys(
    db_path: str,
    main_table: str,
    *,
    key_column: str = "id",
    strategy: str = "nullify",                   #  "nullify" | "repoint_id" | "repoint_identity"
    repoint_to_ids: Optional[Dict[str, int]] = None,      # {"fk_col": 123}
    repoint_from_identity: Optional[Dict[str, Mapping[str, Any]]] = None,  # {"fk_col": {"name": "...", ...}}
    # params required when using repoint_identity
    table_columns: Optional[Dict[str, List[str]]] = None,
    identity_map: Optional[Dict[str, List[str]]] = None,
    on_missing_identity: str = "skip",         # for resolve_or_create_fk
    allow_extra_attrs: bool = True,
    existing_policy: str = "keep",
    mutable_extras: Optional[Dict[str, List[str]]] = None,
    # selection / I/O
    filters: Optional[dict] = None,
    dry_run: bool = True,
    output_dir: str = "../data/update_previews",
):
    """
    
    Detect invalid FKs in main_table and either:
      - nullify them,
      - repoint to a specific ID (repoint_id), or
      - repoint to the ID resolved/created from an identity payload (repoint_identity).

    """
    repoint_to_ids = repoint_to_ids or {}
    repoint_from_identity = repoint_from_identity or {}
    filters = filters or {}

    total_changed = 0
    total_targeted = 0

    # Group run outputs under a single timestamped parent
    run_id = time.strftime("%Y%m%d-%H%M%S")
    parent_out_dir = os.path.join(output_dir, f"{main_table}__fix_invalid_fk__{run_id}")
    os.makedirs(parent_out_dir, exist_ok=True)

    out_dir_last = None
    with sqlite3.connect(db_path) as conn:
        fk_map = get_fk_info(conn, main_table)  # {fk_col: {"table": target, "to_col": "id"}}

        for fk_col, fk in fk_map.items():
            target_table = fk["table"]
            to_col = fk.get("to_col", "id")

            pending_fk_payloads = {}

            # Build subquery: ids whose fk_col is non-null but doesn't exist in target_table
            extra_where = [
                f"{main_table}.{fk_col} IS NOT NULL",
                f"NOT EXISTS (SELECT 1 FROM {target_table} t WHERE t.{to_col} = {main_table}.{fk_col})",
            ]
            where, params, joins = build_query_context(
                main_table=main_table,
                filters=filters,
                base_tables={main_table},
                extra_where=extra_where,
            )
            sub = f"SELECT {main_table}.{key_column} FROM {main_table} " + " ".join(joins)
            if where:
                sub += " WHERE " + " AND ".join(where)

            # Decide update_dict per strategy
            if strategy == "nullify":
                update_dict = {fk_col: None}

            elif strategy == "repoint_id":
                if fk_col not in repoint_to_ids:
                    logger.info(f"[SKIP] No repoint_to_ids configured for {fk_col}")
                    continue
                update_dict = {fk_col: int(repoint_to_ids[fk_col])}

            elif strategy == "repoint_identity":
                # Must have identity payload & the metadata needed by resolve_or_create_fk
                identity_payload = repoint_from_identity.get(fk_col)
                if not identity_payload:
                    logger.info(f"[SKIP] No identity payload for {fk_col}")
                    continue
                if identity_map is None or table_columns is None:
                    raise ValueError("repoint_identity requires identity_map and table_columns.")
                # Resolve/create the target row to get an ID
                fk_result = resolve_or_create_fk(
                    conn=conn,
                    target_table=target_table,
                    row=pd.Series(identity_payload),
                    table_columns=table_columns,
                    identity_map=identity_map,
                    on_missing_identity=on_missing_identity,
                    allow_extra_attrs=allow_extra_attrs,
                    existing_policy=existing_policy,
                    mutable_extras=mutable_extras,
                    dry_run=dry_run,
                    dry_run_id_seed=-1
                )
                if isinstance(fk_result, tuple):
                    new_id, payload = fk_result
                    pending_fk_payloads[(target_table, new_id)] = payload
                    update_dict = {fk_col: int(new_id)}

                else:
                    new_id = fk_result
                    if new_id is None:
                        logger.info(f"[SKIP] Could not resolve/create identity for {fk_col} → leaving unchanged")
                        continue
                    update_dict = {fk_col: int(new_id)}

            else:
                raise ValueError("strategy must be 'nullify', 'repoint_id', or 'repoint_identity'.")

            # Unique preview folder per FK column
            col_out_dir = os.path.join(parent_out_dir, f"{fk_col}")
            os.makedirs(col_out_dir, exist_ok=True)

            out_dir_last, n_changed, n_targeted = execute_update(
                db_path,
                main_table=main_table,
                key_column=key_column,
                update_dict=update_dict,
                id_subquery_sql=sub,
                id_subquery_params=params,
                preview_output_dir=col_out_dir,
                dry_run=dry_run,
                pending_fk_payloads=pending_fk_payloads,
            )

            total_changed += n_changed
            total_targeted += n_targeted

    return parent_out_dir, total_changed, total_targeted

# --------------------------------------------------------------
#                                  Update all records
#--------------------------------------------------------------
def update_all_records(
    db_path: str,
    main_table: str,
    update_dict: dict[str, object],
    *,
    key_column: str = "id",
    dry_run: bool = True,
    output_dir: str = "../data/update_previews",
    # Optional: let this also repoint FKs from natural keys if you want
    allow_fk_resolution: bool = True,
):
    """
    Update every row in main_table with values in update_dict.
    If allow_fk_resolution=True, natural FK specs are converted to *_id first.
    """
    with sqlite3.connect(db_path) as conn:
        fk_map = get_fk_info(conn, main_table)
        schema = get_multi_table_schema(conn, main_table, fk_map)
        pending_fk_payloads = {}
        effective_update = dict(update_dict)
        if allow_fk_resolution:
            effective_update, pending_fk_payloads = preprocess_update_dict_for_fk(
                conn=conn,
                main_table=main_table,
                update_dict=effective_update,
                table_relationships=TABLE_RELATIONSHIPS,
                identity_map=IDENTITY_MAP,
                table_columns=schema,
                on_missing_identity="error",
                allow_extra_attrs=True,
                identity_aliases=IDENTITY_ALIASES,
                dry_run=dry_run,
                dry_run_id_seed=-1,
                pending_fk_payloads=pending_fk_payloads
            )

        id_sub = f"SELECT {main_table}.{key_column} FROM {main_table}"
        return execute_update(
            db_path,
            main_table=main_table,
            key_column=key_column,
            update_dict=effective_update,
            id_subquery_sql=id_sub,
            id_subquery_params={},
            preview_output_dir=output_dir,
            dry_run=dry_run,
            pending_fk_payloads=pending_fk_payloads
        )

# -----------------------------------------------------------------
#                 Update codes with invalid categorical values
#------------------------------------------------------------------

def update_invalid_categorical_values(
    db_path: str,
    table: str,
    *,
    column: str,
    allowed_values: set[str],
    default_for_unknown: str = "unknown",
    case_insensitive: bool = True,
    filters: dict | None = None,
    key_column: str = "id",
    dry_run: bool = True,
    output_dir: str = "../data/update_previews",
    chunk_size: int = 400
):
    """
    Normalize a TEXT categorical column to a canonical set of allowed literals.

    Rules:
      1) If current value is exactly in `allowed_values` -> keep as is.
      2) Else if LOWER(TRIM(current)) matches LOWER(TRIM(any allowed)) -> set to that canonical allowed literal.
      3) Else -> set to `default_for_unknown` (e.g., "unknown").

    Behavior:
      - Writes ONE preview CSV containing ALL table columns + 'new_value' and 'change_<column>' ("old -> new").
      - On dry_run=True: no DB changes, only the CSV.
      - On dry_run=False: applies only truly changed rows (grouped by target value for simplicity/efficiency).

    Returns:
      (out_dir: str | Path, n_changed: int, n_targeted: int)
    """
    import sqlite3
    import pandas as pd
    from datetime import datetime
    from pathlib import Path

    filters = filters or {}

    # -- Build canonical normalization function & map of normalized allowed -> canonical literal
    def _norm(s: str | None) -> str | None:
        if s is None:
            return None
        s2 = s.strip()
        return s2.lower() if case_insensitive else s2

    # Map normalized allowed -> canonical spelling from allowed_values
    allowed_norm_to_canonical: dict[str, str] = {}
    for lit in allowed_values:
        n = _norm(lit)
        if n is not None and n != "":
            allowed_norm_to_canonical[n] = lit  # prefer provided canonical spelling

    with sqlite3.connect(db_path) as conn:
        # Validate columns exist
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        cols = {r[1] for r in cur.fetchall()}
        if column not in cols:
            raise ValueError(f"Column '{column}' not found in table '{table}'. Columns={sorted(cols)}")
        if key_column not in cols:
            raise ValueError(f"Key column '{key_column}' not found in table '{table}'.")

        # Build query (JOIN-aware) for candidate rows: non-NULL & non-empty in target column
        col_ref = f"{table}.{column}"
        base_where = [f"{col_ref} IS NOT NULL", f"LENGTH(TRIM({col_ref})) > 0"]

        where, params, joins = build_query_context(
            main_table=table,
            filters=filters,
            base_tables={table},
            extra_where=base_where
        )

        # Select ALL columns for preview context
        q = f"SELECT {table}.* FROM {table} " + (" ".join(joins) if joins else "")
        if where:
            q += " WHERE " + " AND ".join(where)

        df = pd.read_sql_query(q, conn, params=params)

        # Prepare output folder
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(output_dir) / f"updates_{table}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if df.empty:
            # Nothing to review
            preview_path = out_dir / f"{table}_categorical_preview.csv"
            # Write empty with headers at least
            empty_cols = list(cols) + ["new_value", f"change_{column}"]
            pd.DataFrame(columns=empty_cols).to_csv(preview_path, index=False)
            return str(out_dir), 0, 0

        # Compute new values
        def compute_new(old_val: str | None) -> str:
            if old_val is None or str(old_val).strip() == "":
                return default_for_unknown
            s = str(old_val)
            # Rule 1: exact literal in allowed -> keep as-is
            if s in allowed_values:
                return s
            # Rule 2: normalized match -> canonical allowed literal
            n = _norm(s)
            if n in allowed_norm_to_canonical:
                return allowed_norm_to_canonical[n]
            # Rule 3: fallback
            return default_for_unknown

        df["__old__"] = df[column]
        df["new_value"] = df["__old__"].apply(compute_new)

        # Determine which rows actually change (string compare; treat None vs "None" carefully)
        def _different(a, b) -> bool:
            if a is None and b is None:
                return False
            return str(a) != str(b)

        df[f"change_{column}"] = df.apply(
            lambda r: f"{r['__old__']} -> {r['new_value']}" if _different(r["__old__"], r["new_value"]) else None,
            axis=1
        )

        # Only changed rows are targets
        changed_mask = df[f"change_{column}"].notna()
        changed_df = df.loc[changed_mask].copy()
        n_changed = len(changed_df)
        n_targeted = len(df)  # rows examined under current filters

        # Always write ONE preview CSV with full table + new_value + change_<column>
        preview_cols = list(df.columns)  # already includes all table columns plus helper columns
        # Remove internal __old__ from the CSV (we have change_ and new_value already)
        preview_cols = [c for c in preview_cols if c != "__old__"]
        # Ensure new_value and change_<column> are at the end
        base_cols = [c for c in df.columns if c not in ("__old__", "new_value", f"change_{column}")]
        ordered_cols = base_cols + ["new_value", f"change_{column}"]
        df[ordered_cols].to_csv(out_dir / f"{table}_categorical_preview.csv", index=False)

        if dry_run or n_changed == 0:
            return str(out_dir), n_changed, n_targeted

        # --- Apply updates for changed rows only, grouped by target value ---
        # Build groups: new_value -> list of ids
        if key_column not in df.columns:
            raise ValueError(f"Primary key column '{key_column}' not present in selected columns.")

        groups: dict[str, list[int]] = {}
        for _, row in changed_df.iterrows():
            rid = int(row[key_column])
            nv = row["new_value"]
            groups.setdefault(nv, []).append(rid)

        # Execute per group to keep SQL simple
        for new_val, ids in groups.items():
            for i in range(0, len(ids), chunk_size):
                chunk = ids[i:i+chunk_size]
                placeholders = ", ".join(["?"] * len(chunk))
                sql = f"UPDATE {table} SET {column} = ? WHERE {key_column} IN ({placeholders})"
                conn.execute(sql, [new_val] + chunk)
        conn.commit()

        return str(out_dir), n_changed, n_targeted


if __name__ == "__main__":
    DB_PATH = "/Users/masoomeshafiee/Projects/data_organization/data-management-system-SQLite/db/Reyes_lab_data.db" # <-- change this
    
    #db, n_previewd, n_miss = run_updates_from_csv(db_path=DB_PATH,
    #main_table="Experiment",
    #key_column="id",
    #csv_path="/Users/masoomeshafiee/Projects/data_organization/data-management-system-SQLite/data/update_previews/test_update.csv",
    #commit_missing_fk= False,
    #dry_run = True,
    #output_dir="../data/update_previews",
    #updated = update_records_by_filter(DB_PATH, "Experiment", {"is_valid": "Y"}, {"is_valid": "N"}, dry_run=True)
    #update_records_from_csv_relational_strict(DB_PATH, "Experiment", "../data/update_previews/test_update.csv", dry_run=True)
    # identity_cols=["organism_id", "protein_id", "strain_id", "condition_id", "capture_setting_id", "user_id", "date", "replicate"]
    update_dict = {"capture_type":"test_las", "time_interval":"200", "exposure_time":"50", "dye_concentration_value":"50", "dye_concentration_unit":"nM", "fluorescent_dye":"test2", "laser_wavelength":"999", "laser_intensity":"10", "camera_binning":"2", "objective_magnification":"60", "pixel_size":"100"}
    out_dir, n_changed, n_targeted = update_records_by_filter(DB_PATH,main_table="Experiment",update_dict=update_dict,filters={"condition":"cpt"},dry_run=True)

    #parent_out_dir, per_column_dirs, n_changed, n_targeted = update_missing_values_with_default(DB_PATH, main_table="Experiment", column_defaults={"experiment_path": "will be", "comment": "TO BE"}, filters = {"condition":"cpt"}, dry_run=True)
    #parent_out_dir, per_column_dirs, n_changed, n_targeted = update_missing_values_with_default(DB_PATH, main_table="CaptureSetting", column_defaults={"pixel_size":"100"}, filters = {}, dry_run=True)
    #table_columns = get_multi_table_schema(sqlite3.connect(DB_PATH), "Experiment", get_fk_info(sqlite3.connect(DB_PATH), "Experiment"))
    #parent_out_dir, total_changed, total_targeted = update_invalid_foreign_keys(DB_PATH, main_table="Experiment", strategy="repoint_identity",repoint_from_identity ={"capture_setting_id":{"capture_type":"fast", "exposure_time":'0.01', "time_interval":"0.01", "dye_concentration_value":"50", "laser_intensity":"60%"}},table_columns = table_columns,
    #identity_map = IDENTITY_MAP,         # for resolve_or_create_fk
    #allow_extra_attrs = True,
    #existing_policy ="update",
    # selection / I/O
    #mutable_extras=MUTABLE_EXTRAS, filters={"condition":"untreated"}, dry_run=True) # Note: if the filter itself is orphan, it cant find the records to update. ex: if the filter is condition:" " --> Since it does not exist in the condition table, the filter query returns no record to even check the orphand for capture setting table. 
    
    
    #out_dir, n_changed, n_targeted = update_all_records(DB_PATH,main_table="Experiment",update_dict={"condition":"cpt", 'concentration_value':"69", "concentration_unit":"um"},dry_run=True)

    #out_dir, n_changed, n_targeted = update_invalid_categorical_values(DB_PATH, "CaptureSetting",column="dye_concentration_unit",allowed_values=ALLOWED_CATEGORICALS["dye_concentration_unit"],default_for_unknown="unknown",dry_run=True)

    print(out_dir, n_changed, n_targeted)
