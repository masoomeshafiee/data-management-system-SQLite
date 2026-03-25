from pathlib import Path
from datetime import datetime
import sqlite3
import pandas as pd
from queries import build_query_context, fetch_rows_by_ids, fetch_unique_conflicts
from data_validation import validate_domain_constraints, validate_fk_constraints, validate_unique_constraints
from update_records import get_fk_info
import os
import logging

# ----------------------- Logging setup -----------------------
logger = logging.getLogger("update_preflight")
if not logger.handlers:
    fh = logging.FileHandler("../data/update_preflight.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)




def preprocess_update(
    db_path: str,
    *,
    main_table: str,
    key_column: str="id",
    # How to select targets:
    id_list: list[int] | None = None,
    id_subquery_sql: str | None = None,
    id_subquery_params: dict | tuple | None = None,
    filters: dict | None = None,
    # Function that propose new values per row (row -> dict of {col: new_val})
    proposer = None,
    # Validation hooks:
    domain_validators: dict[str, callable] | None = None,   # col -> fn(value)->bool
    fk_constraints: dict[str, str] | None = None,   # {"organism_id": "Organism", ...}
    unique_identity: dict[str, list[str]] | None = None,
    # IO:
    output_dir: str = "../data/update_previews",
    chunk_size: int = 400):

    """
    Preflight report for an update on `table`.
    Returns (out_dir, report_dict) and writes CSVs:
      - <table>_preflight_preview.csv  (all cols + new_<col> + change_<col>)
      - <table>_preflight_conflicts.csv (if any uniqueness conflicts)
      - <table>_preflight_domain_errors.csv (if any)
      - <table>_preflight_fk_errors.csv (if any)
    """
    filters = filters or {}
    update_dict_per_row = update_dict_per_row or {}
    domain_validators = domain_validators or {}
    fk_constraints = fk_constraints or {}
    unique_identity = unique_identity or {}

    # --- Resolve targets
    conn = sqlite3.connect(db_path)

    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        cur = conn.cursor()

        # Validate table/columns existence as we go
        cur.execute(f"PRAGMA table_info({main_table});")
        table_cols = {row[1] for row in cur.fetchall()}

        if key_column not in table_cols:
            logger.error(f"Key column '{key_column}' not found in table '{main_table}'")
            raise ValueError(f"Key column '{key_column}' not found in table '{main_table}'")
        
        if id_list is None:
            if not id_subquery_sql:
                # Build filter-based query
                where, params, joins = build_query_context(main_table=main_table, filters=filters, base_tables={main_table})
                query = f"SELECT {main_table}.{key_column} FROM {main_table}" + " ".join(joins)
                if where:
                    query += " WHERE " + " AND ".join(where)
                id_df = pd.read_sql_query(query, conn, params=params)
            else:
                # Use provided subquery
                id_df = pd.read_sql_query(id_subquery_sql, conn, params=id_subquery_params or {})
            id_list = id_df.iloc[:,0].dropna().astype(int).tolist()
        logger.info(f"Resolved {len(id_list)} target rows in table '{main_table}' for update preflight.")

        if not id_list:
            logger.warning("No target rows found for the update operation.")
            return str(output_dir), {
                "n_targeted": 0,
                "n_changed": 0,
                "conflicts": pd.DataFrame(),
                "domain_errors": pd.DataFrame(),
                "fk_errors": pd.DataFrame(),
            }
        # --- Fetch current data in chunks
        rows = []
        for i in range(0, len(id_list), chunk_size):
            chunk = id_list[i:i+chunk_size]
            placeholders = ", ".join(["?"] * len(chunk))
            q = f"SELECT * FROM {main_table} WHERE {key_column} IN ({placeholders})"
            rows.append(pd.read_sql_query(q, conn, params=chunk))
        df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        
        # proposed changes
        # proposer(row: pd.Series) -> dict {col: new_value}
        proposed_cols = set()
        proposals = []
        for _, r in df.iterrows():
            d = proposer(r) if proposer else {}
            if not isinstance(d, dict):
                logger.error("Proposer function must return a dict[col->new_val]")
                raise ValueError("proposer must return a dict[col->new_val]")
            proposed_cols |= set(d.keys())
            proposals.append(d)
        df["_proposed"] = proposals

        # --- Domain checks
        domain_errors = []
        for idx, r in df.iterrows():
            prop = r["_proposed"]
            for col, fn in domain_validators.items():
                if col in prop:
                    try:
                        ok = fn(prop[col])
                    except Exception as e:
                        ok = False
                    if not ok:
                        domain_errors.append({
                            key_column: r[key_column],
                            "column": col,
                            "proposed_value": prop[col],
                            "error": "domain_validation_failed"
                        })
        domain_errors_df = pd.DataFrame(domain_errors)
        if not domain_errors_df.empty:
            logger.info(f"Found {len(domain_errors_df)} domain validation errors. Pleaeese review the preflight report.")

        # ---------- AUTO FK EXISTENCE CHECKS ----------
        fk_map = get_fk_info(conn, main_table)  # <-- your existing helper
        # Only check FK columns that are actually part of proposals
        fk_cols_to_check = [fk_col for fk_col in fk_map.keys() if fk_col in proposed_cols and fk_col in table_cols]

        fk_errors = []
        for fk_col in fk_cols_to_check:
            parent_table = fk_map[fk_col]["table"]
            to_col = fk_map[fk_col].get("to_col", "id")

            # Collect all proposed new FK ids for this column
            new_fk_ids = sorted({
                int(v) for prop in proposals for (c, v) in prop.items()
                if c == fk_col and v is not None
            })
            if not new_fk_ids:
                continue

            placeholders = ", ".join(["?"] * len(new_fk_ids))
            exist_ids = pd.read_sql_query(
                f"SELECT {to_col} AS id FROM {parent_table} WHERE {to_col} IN ({placeholders})",
                conn, params=new_fk_ids
            )["id"].astype(int).tolist()
            exist_set = set(exist_ids)
            missing = [vid for vid in new_fk_ids if vid not in exist_set]

            for m in missing:
                fk_errors.append({
                    "fk_column": fk_col,
                    "parent_table": parent_table,
                    "missing_id": m
                })

        fk_errors_df = pd.DataFrame(fk_errors)
        if not fk_errors_df.empty:
            logger.info(f"Found {len(fk_errors_df)} foreign key errors. Please review the preflight report.")
            fk_errors_df.to_csv(output_dir / f"{main_table}_preflight_fk_errors.csv", index=False)

    # ---------- UNIQUENESS CONFLICTS (optional) ----------
        conflicts_df = pd.DataFrame()
        if unique_keys:
            df_proj = df.copy()
            # project proposed values where they change
            for col in proposed_cols:
                proj_col = f"__proj__{col}"
                df_proj[proj_col] = df_proj[f"new_{col}"].where(df_proj[f"change_{col}"].notna(), df_proj[col])

            frames = []
            for key_cols in unique_keys:
                proj_cols = [(f"__proj__{c}" if c in proposed_cols else c) for c in key_cols]
                gb = df_proj.groupby(proj_cols, dropna=False, as_index=False, sort=False).size().rename(columns={"size":"grp_size"})
                labeled = df_proj.merge(gb, on=proj_cols, how="left")
                conf = labeled[labeled["grp_size"] > 1].copy()
                if not conf.empty:
                    conf["__unique_key__"] = ",".join(key_cols)
                    conf["__group_key__"] = conf[proj_cols].astype(str).agg("|".join, axis=1)
                    frames.append(conf)
            if frames:
                conflicts_df = pd.concat(frames, ignore_index=True)
                keep_cols = [key_column, "__unique_key__", "__group_key__", "grp_size"]
                base_cols = [c for c in df.columns if not (c.startswith("new_") or c.startswith("change_") or c == "_proposed")]
                tail_cols = [c for c in df.columns if (c.startswith("new_") or c.startswith("change_"))]
                cols = [c for c in keep_cols + base_cols + tail_cols if c in conflicts_df.columns]
                conflicts_df[cols].sort_values(["__unique_key__", "__group_key__", key_column]).to_csv(
                    out_dir / f"{table}_preflight_conflicts.csv", index=False
                )









