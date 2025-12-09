import sqlite3
import csv
import os
import re
from pathlib import Path
import logging

logging.basicConfig(
    filename="/Users/masoomeshafiee/Desktop/Presentation/db_import.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def normalize_value(v):
    if isinstance(v, str):
        return v.strip().lower()
    try:
        return  round(float(v), 5)
    except (TypeError, ValueError):
        return v


def get_or_create_id(cursor, table, unique_fields: dict):
    """Return id of existing row or insert a new one."""
    placeholders = ' AND '.join(f"{k}=?" for k in unique_fields)
    values = tuple(unique_fields.values())
    cursor.execute(f"SELECT id FROM {table} WHERE {placeholders}", values)
    result = cursor.fetchone()
    if result:
        return result[0]

    columns = ', '.join(unique_fields)
    placeholders = ', '.join('?' for _ in unique_fields)
    cursor.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", values
    )
    logging.info(f"Inserted new value in the {table} table  with values: {unique_fields}")
    return cursor.lastrowid

def get_experiment_id_from_signature(cursor, signature):
    """
    Look up Experiment.id directly from a signature string in linked_experiment_signatures.
    Signature format in CSV: YYYYMMDD|replicate|organism|protein|condition|capture_type
    """
    signature = signature.lower()
    date, replicate, organism, protein, condition, capture_type = signature.split("|")
    replicate = int(replicate)
    cursor.execute("""
                    SELECT e.id AS experiment_id
                    FROM Experiment e
                    JOIN Organism o ON e.organism_id = o.id
                    JOIN Protein p ON e.protein_id = p.id
                    JOIN Condition c ON e.condition_id = c.id
                    JOIN CaptureSetting cs ON e.capture_setting_id = cs.id
                    WHERE o.organism_name = ?
                      AND p.protein_name = ?
                      AND c.condition_name = ?
                      AND cs.capture_type = ?
                      AND e.date = ?
                      AND e.replicate = ?
                """, (organism, protein, condition, capture_type, date, replicate))
    #if cursor.fetchmany():
    #    print(f"Multiple experiments found for signature: {signature}. Please refine your search.")
    #   return
    results = cursor.fetchall()
    if not results:
        logging.warning(f"No matching experiment for signature: {signature}")
        return None
    elif len(results) > 1:
        logging.warning(f"Multiple experiments found for signature: {signature}")
        cursor.execute("""
                    SELECT e.id AS experiment_id, e.date, e.replicate, e.comment, e.is_valid, cs.capture_type, cs.id as capture_id, cs.time_interval
                    FROM Experiment e
                    JOIN Organism o ON e.organism_id = o.id
                    JOIN Protein p ON e.protein_id = p.id
                    JOIN Condition c ON e.condition_id = c.id
                    JOIN CaptureSetting cs ON e.capture_setting_id = cs.id
                    WHERE o.organism_name = ?
                      AND p.protein_name = ?
                      AND c.condition_name = ?
                      AND cs.capture_type = ?
                      AND e.date = ?
                      AND e.replicate = ?
                """, (organism, protein, condition, capture_type, date, replicate))
        rows = cursor.fetchall()
        print(f"Experiments: { [row for row in rows]}")
        #print("I was here")
        return None
    
    return results[0][0]


def get_or_create_experiment(cursor, row):

    pass
        
'''
def get_or_create_id(cursor, table, unique_fields: dict):
    # Normalize all values
    normalized_fields = {k: normalize_value(v) for k, v in unique_fields.items()}

    # Build SQL placeholders using normalized field names and values
    conditions = []
    values = []

    for k, v in normalized_fields.items():
        if isinstance(v, str):
            conditions.append(f"LOWER(TRIM({k})) = ?")
            values.append(v)
        else:
            conditions.append(f"{k} = ?")
            values.append(v)

    where_clause = ' AND '.join(conditions)
    cursor.execute(f"SELECT id FROM {table} WHERE {where_clause}", values)
    result = cursor.fetchone()

    if result:
        return result[0]

    # Insert original (non-normalized) values
    columns = ', '.join(unique_fields.keys())
    placeholders = ', '.join('?' for _ in unique_fields)
    cursor.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
        tuple(unique_fields.values())
    )
    return cursor.lastrowid


'''
def insert_from_csv(csv_path, db_path, skipped_rows):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Optional: skip invalid rows
            # skip the empty rows
            if not any(row.values()):
                logging.info("Skipped empty row {row}")
                continue
            if row["file_category"] == "raw":
                # --- Foreign key lookups ---
                organism_id = get_or_create_id(cursor, "Organism", {"organism_name": normalize_value(row["organism"])})
                protein_id = get_or_create_id(cursor, "Protein", {"protein_name": normalize_value(row["protein"])})
                strain_id = get_or_create_id(cursor, "StrainOrCellLine", {"strain_name": normalize_value(row["strain"])})
                condition_id = get_or_create_id(cursor, "Condition", {
                    "condition_name": normalize_value(row["condition"]),
                    "concentration_value":normalize_value(row["concentration_value"]),
                    "concentration_unit": normalize_value(row["concentration_unit"])
                })
                user_id = get_or_create_id(cursor, "User", {
                    "name": normalize_value(row["user_name"]),
                    "last_name": normalize_value(row["user_last_name"]),
                    "email": normalize_value(row["user_email"])
                })

                # --- CaptureSetting ---
                capture_setting_id = get_or_create_id(cursor, "CaptureSetting", {
                    "capture_type": normalize_value(row["capture_type"]),  # You might extract this from context later
                    "exposure_time": normalize_value(row["exposure_time"]),
                    "time_interval": normalize_value(row["time_interval"]),
                    "fluorescent_dye": normalize_value(row["fluorescent_dye"]),
                    "dye_concentration_value": normalize_value(row["dye_concentration_value"]),
                    "dye_concentration_unit": normalize_value(row["dye_concentration_unit"]),
                    "laser_wavelength": row["laser_wavelength"],
                    "laser_intensity": normalize_value(row["laser_intensity"]),
                    "camera_binning": row["camera_binning"],
                    "objective_magnification":normalize_value(row["objective_magnification"]),
                    "pixel_size": row["pixel_size"]
                })

                # --- Experiment ---
                experiment_lookup = {
                    "organism_id": organism_id,
                    "protein_id": protein_id,
                    "strain_id": strain_id,
                    "condition_id": condition_id,
                    "capture_setting_id": capture_setting_id,
                    "user_id": user_id,  # Assumes file_path is representative
                    "date": row["date"],
                    "replicate": row.get("replicate"),
                    "is_valid": row.get("is_valid", ""),  # Default to valid if not specified
                    "comment": row.get("comment",""),  # Optional comment
                    "experiment_path": row.get("experiment_path", "")  # Optional path
                }
                # Only use the UNIQUE constraint columns
                unique_keys = ["organism_id", "protein_id", "strain_id", "condition_id",
               "capture_setting_id", "user_id", "date", "replicate"]

                # Check if experiment already exists
                placeholders = ' AND '.join(f"{k}=?" for k in unique_keys)
                values = tuple(experiment_lookup[k] for k in unique_keys)
                cursor.execute(f"SELECT id FROM Experiment WHERE {placeholders}", values)
                exp_result = cursor.fetchone()

                #logging.debug(f"Experiment lookup keys: {experiment_lookup}")
                #logging.debug(f"Experiment lookup values: {values}")

                if exp_result:
                    experiment_id = exp_result[0]
                else:
                    columns = ', '.join(experiment_lookup)
                    placeholders = ', '.join('?' for _ in experiment_lookup)
                    values = tuple(experiment_lookup.values())
                    try:
                        cursor.execute(
                            f"INSERT INTO Experiment ({columns}) VALUES ({placeholders})",
                            values
                        )
                        experiment_id = cursor.lastrowid
                        logging.info(f"Inserted new experiment with values: {experiment_lookup}")
                    except sqlite3.IntegrityError as e:
                        logging.error(f"Duplicate experiment. insert failed: {experiment_lookup} -> {e}")
                        row_with_reason = row.copy()
                        row_with_reason["skip_reason"] = f"Duplicate experiment. insert failed: {e}"
                        skipped_rows.append(row_with_reason)
                        continue

            else:
                # If not a raw file, we assume the experiment_id is already known YYYYMMDD|replicate|organism|protein|condition|capture_type
                experiment_signature = "|".join([
                    row["date"],  # YYYYMMDD
                    str(row.get("replicate", 1)),  # Default to 1 if not provided
                    normalize_value(row["organism"]),
                    normalize_value(row["protein"]),
                    normalize_value(row["condition"]),
                    normalize_value(row["capture_type"])
                ])
                experiment_id = get_experiment_id_from_signature(cursor, experiment_signature)
                if not experiment_id:
                    logging.warning(f"No experiment found for signature: {experiment_signature} for the file {row['file_name']}. Skipping row.")
                    row_with_reason = row.copy()
                    row_with_reason["skip_reason"] = f"No experiment found for signature: {experiment_signature}"
                    skipped_rows.append(row_with_reason)
                    continue
                
            # --- Insert into RawFile ---
            if row["file_category"] == "raw":
                cursor.execute("""
                    SELECT id FROM RawFiles WHERE experiment_id=? AND file_name=? AND field_of_view=? AND file_type=?
                """, (experiment_id, row["file_name"], row["field_of_view"], row.get("file_type")))
                result = cursor.fetchone()
                if result:
                    logging.info(f"Duplicate raw file skipped: {row['file_name']}")
                    row_with_reason = row.copy()
                    row_with_reason["skip_reason"] = "Duplicate raw file"
                    skipped_rows.append(row_with_reason)
                else:
                    cursor.execute("""
                        INSERT INTO RawFiles (experiment_id, file_name, field_of_view, file_type, file_path)
                        VALUES (?, ?, ?, ?, ?)
                    """, (experiment_id, row["file_name"], row["field_of_view"], row.get("file_type"), row["file_path"]))
                    logging.info(f"Inserted raw file: {row['file_name']}")
                

            
            # --- Insert into TrackingFiles ---
            if row["file_category"] == "tracking":
                cursor.execute("""
                    SELECT id FROM TrackingFiles WHERE experiment_id=? AND file_name=? AND field_of_view=? AND file_type=? AND threshold=? AND linking_distance=? AND gap_closing_distance=? AND max_frame_gap=?
                """, (experiment_id, row["file_name"], row["field_of_view"], row["file_type"], row["threshold"], row["linking_distance"], row["gap_closing_distance"], row["max_frame_gap"]))
                result = cursor.fetchone()
                if result:
                    logging.info(f"Duplicate tracking file skipped: {row['file_name']}")
                    row_with_reason = row.copy()
                    row_with_reason["skip_reason"] = "Duplicate tracking file"
                    skipped_rows.append(row_with_reason)
                else:
                    cursor.execute("""
                        INSERT INTO TrackingFiles (experiment_id, file_name, field_of_view, file_type, file_path, threshold, linking_distance, gap_closing_distance, max_frame_gap)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (experiment_id, row["file_name"], row["field_of_view"], row["file_type"], row["file_path"], row["threshold"], row["linking_distance"], row["gap_closing_distance"], row["max_frame_gap"]))
                    logging.info(f"Inserted tracking file: {row['file_name']}")
                

                
            # --- Insert into Masks ---
            if row["file_category"] == "mask":
                cursor.execute("""
                    SELECT id FROM Masks WHERE experiment_id=? AND mask_name=? AND field_of_view=? AND mask_type=? AND file_type=? AND segmentation_method=?
                """, (experiment_id, row["file_name"], row["field_of_view"], row["mask_type"].strip(), row["file_type"], row["segmentation_method"]))
                result = cursor.fetchone()
                if result:
                    logging.info(f"Duplicate mask skipped: {row['file_name']}")
                    row_with_reason = row.copy()
                    row_with_reason["skip_reason"] = "Duplicate mask"
                    skipped_rows.append(row_with_reason)
                else:
                    cursor.execute("""
                        INSERT INTO Masks (experiment_id, mask_name, field_of_view, mask_type, file_type, mask_path, segmentation_method, segmentation_parameters)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (experiment_id, row["file_name"], row["field_of_view"], row["mask_type"].strip(), row["file_type"], row["file_path"], row["segmentation_method"], row["segmentation_parameters"]))
                    logging.info(f"Inserted mask: {row['file_name']}")


            # --- Insert into AnalysisFiles + link table ---
            if row["file_category"] == "analysis_file":
                # Get or create AnalysisFile
                cursor.execute("""
                    SELECT id FROM AnalysisFiles WHERE file_name=? AND field_of_view=? AND file_type=?
                """, (row["file_name"], row["field_of_view"], row["file_type"]))
                res = cursor.fetchone()
                if res:
                    analysis_file_id = res[0]
                    logging.info(f"Duplicate analysis file skipped: {row['file_name']}")
                    row_with_reason = row.copy()
                    row_with_reason["skip_reason"] = "Duplicate analysis file"
                    skipped_rows.append(row_with_reason)
                else:
                    cursor.execute("""
                        INSERT INTO AnalysisFiles (file_name, file_path, file_type, field_of_view)
                        VALUES (?, ?, ?, ?)
                    """, (row["file_name"], row["file_path"], row["file_type"], row["field_of_view"]))
                    analysis_file_id = cursor.lastrowid
                    logging.info(f"Inserted analysis file: {row['file_name']}")

                # Link to experiment
                cursor.execute("""
                    INSERT OR IGNORE INTO ExperimentAnalysisFiles (experiment_id, analysis_file_id)
                    VALUES (?, ?)
                """, (experiment_id, analysis_file_id))
                if cursor.rowcount > 0:
                    logging.info(f"Linked analysis file {row['file_name']} to experiment ID {experiment_id}")
                else:
                    logging.info(f"Link between analysis file {row['file_name']} and experiment ID {experiment_id} already exists")
                    row_with_reason = row.copy()
                    row_with_reason["skip_reason"] = "Duplicate analysis file-experiment link"
                    skipped_rows.append(row_with_reason)

    conn.commit()
    conn.close()
    print("Database updated successfully.")

# Analysis files and results insertion

def insert_analysis_csv(csv_path, db_path, skipped_analysis_rows):
    """
    Insert analysis metadata from analysis_metadata.csv into:
    - AnalysisFiles
    - ExperimentAnalysisFiles (link table)
    - AnalysisResults (if result fields are filled)
    - AnalysisResultExperiments (link table for results)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not any(row.values()):
                continue
            # *********************************************************
            '''
            # Insert into AnalysisFiles
            conn.execute("""
                INSERT INTO AnalysisFiles (file_path, analysis_type)
                VALUES (?, ?)
            """, (
                row["analysis_files_path"],
                row.get("analysis_method") or None
            ))
            analysis_file_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Link AnalysisFiles to Experiments
            signatures = [sig.strip() for sig in row["linked_experiment_signatures"].split(",") if sig.strip()]
            for sig in signatures:
                exp_id = get_experiment_id_from_signature(conn, sig)
                if exp_id:
                    conn.execute("""
                        INSERT INTO ExperimentAnalysisFiles (experiment_id, analysis_file_id)
                        VALUES (?, ?)
                    """, (exp_id, analysis_file_id))
                else:
                    print(f"⚠️ No matching Experiment found for signature: {sig}")
            '''
            # *********************************************************
            # If result fields exist, insert into AnalysisResults and link
            # Check if AnalysisResults already exist
            cursor.execute("""
                SELECT id FROM AnalysisResults
                WHERE Analysis_file_path=? AND result_type=? AND result_value=?
            """, (row["analysis_files_path"], row.get("result_type"), row.get("result_value")))
            res = cursor.fetchone()
            if res:
                result_id = res[0]
                logging.info(f"Duplicate analysis result skipped for file: {row['analysis_files_path']}")
                skipped_analysis_rows.append(row)
            else:
                cursor.execute("""
                    INSERT INTO AnalysisResults (result_type, result_value,
                                                 sample_size, standard_error, Analysis_method,
                                                 analysis_parameters, Analysis_file_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (row.get("result_type"), row.get("result_value"),
                      row.get("sample_size"), row.get("standard_error"),
                      row.get("analysis_method"), row.get("analysis_parameters"),
                      row["analysis_files_path"]))
                result_id = cursor.lastrowid
                logging.info(f"Inserted analysis result for file: {row['analysis_files_path']} with ID {result_id}")

            # Link result to experiments
            signatures = [sig.strip() for sig in row["linked_experiment_signatures"].split(",") if sig.strip()]
            for sig in signatures:
                exp_id = get_experiment_id_from_signature(cursor, sig)
                if exp_id:
                    cursor.execute("""
                        INSERT OR IGNORE INTO AnalysisResultExperiments (analysis_result_id, experiment_id)
                        VALUES (?, ?)
                    """, (result_id, exp_id))
                    if cursor.rowcount > 0:
                        logging.info(f"Linked analysis result ID {result_id} to experiment ID {exp_id}")
                    else:
                        logging.info(f"Link between analysis result ID {result_id} and experiment ID {exp_id} already exists")
                        skipped_analysis_rows.append(row)
                else:
                    logging.warning(f"No experiment found for signature: {sig}. Skipping link.")

    conn.commit()
    conn.close()
    print("Database updated successfully.")


# ------------ Utility to write skipped rows to CSV for review ------------
def write_skipped_to_csv(skipped_rows, filename="skipped_rows.csv"):
    """Save skipped rows into a CSV file."""
    if not skipped_rows:
        return

    file_path = Path(filename)
    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=skipped_rows[0].keys())
        writer.writeheader()
        writer.writerows(skipped_rows)

    logging.info(f"Skipped rows written to {file_path}")



# --- Example Run ---
if __name__ == "__main__":
    update_analysis = False  # Set to True if you have analysis metadata to insert
    CSV_PATH = "/Users/masoomeshafiee/Desktop/Presentation/3.metadata_complete.csv"
    ANALYSIS_CSV_PATH = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/dwell time/UV/analysis_metadata.csv"
    DB_PATH = "/Users/masoomeshafiee/Projects/data_organization/data-management-system-SQLite/db/Reyes_lab_data.db"
    skipped_rows = []
    skipped_analysis_rows = []
    insert_from_csv(CSV_PATH, DB_PATH, skipped_rows)
    if update_analysis:
        # If you have an analysis metadata CSV, insert it
        # ANALYSIS_CSV_PATH = "/path/to/your/analysis_metadata.csv"
        # Make sure to set the correct path for your analysis metadata CSV
        insert_analysis_csv(ANALYSIS_CSV_PATH, DB_PATH, skipped_analysis_rows)
        write_skipped_to_csv(skipped_analysis_rows, "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/confocal/skipped_analysis_rows.csv")

    write_skipped_to_csv(skipped_rows, "/Users/masoomeshafiee/Desktop/Presentation/skipped_rows.csv")
    logging.info("Data insertion process completed.")

