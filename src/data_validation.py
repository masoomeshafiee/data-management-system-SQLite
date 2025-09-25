import sqlite3
import csv
import os
import re
import logging
from datetime import datetime
# Setup logging
log_file_path = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/confocal/validation.log"
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w"    # Overwrite each run ("a" for append mode)
)



logging.info("Validation script started")

ALLOWED_NA = {"NA", "N/A"}  # global set

REQUIRED_FIELDS = {
    "raw": ["file_name", "file_type", "date", "replicate", "organism", "protein", "strain", "condition", "capture_type"],
    "tracking": ["file_name", "file_type", "threshold"],
    "mask": ["file_name", "mask_type", "file_type", "segmentation_method"],
    "analysis_file": ["file_name", "file_type"]
}

# Allowed extensions
VALID_EXTENSIONS = {"tif", "tiff", "TIF", "csv", "nd", "npy", "png", "txt", "json", "mat", "pickle", "svg", "pdf", "xlsx"}

# Allowed capture types
CAPTURE_TYPES = {"confocal", "fast", "long"}

# Allowed organisms
ALLOWED_ORGANISMS = {"human", "yeast", "E.coli"}

# Allowed mask types
MASK_TYPES = {"cell", "nucleus", "nucleus-g1", "membrane", "cytoplasm"}

# Allowed dye concentration units
DYE_CONCENTRATION_UNITS = {"pM","nM", "uM", "mM", "M"}

# Allowed condition units
CONDITION_UNITS = {"nM", "uM", "mM", "M", "%", "mJ/m2", "mJ/cm2", "J/cm2", "J/m2"}

# numeric fields per category
NUMERIC_FIELDS = {
    "raw": ["concentration_value", "exposure_time", "time_interval", "dye_concentration_value", "camera_binning", "pixel_size"],
    "tracking": ["threshold", "gap_closing_distance", "linking_distance", "max_frame_gap"],
    "mask": [],
    "analysis_file": []
}

# Regex for filename
FILENAME_PATTERN = re.compile(
    r"""
    ^(?P<date>\d{8})-             # YYYYMMDD
    (?P<replicate>\d+)_         # replicate (2 digits)
    (?P<organism>[A-Za-z]+)_      # organism
    (?P<protein>[A-Za-z0-9]+)_    # protein
    (?P<condition>[A-Za-z0-9]+)_  # condition
    (?P<capture>[A-Za-z0-9]+)_    # capture type
    (?P<fov>\d+)_                 # field of view (integer)
    (?P<filetype>[A-Za-z0-9_\-]+)    # filetype
    (?P<ext>\.[A-Za-z0-9]+)$      # extension
    """,
    re.VERBOSE,
)

# ---------- for the analysis metadata validation -----------

VALID_RESULT_TYPES = {"intensity_over_time", "dwell_time", "bound_fraction", "copy_number"}

VALID_ANALYSIS_METHODS = ["Bound2Learn", "MSD", "SMOL", "RadiusOfGyration", "SingleCellIntensity", "IntensityOverTime"]

def validate_filename(filename: str) -> dict:
    """Validate filename format and return parsed parts if valid."""
    # Validate the pattern of the filename
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return {"valid": False, "error": "Filename does not match required pattern"}

    parts = match.groupdict()

    # Validate date
    year = int(parts["date"][:4])
    month = int(parts["date"][4:6])
    day = int(parts["date"][6:8])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return {"valid": False, "error": "Invalid date in filename"}

    # Validate replicate (must be >=01)
    if int(parts["replicate"]) < 1:
        return {"valid": False, "error": "Replicate must be >=01"}

    # Validate capture type
    if parts["capture"].lower() not in CAPTURE_TYPES:
        return {"valid": False, "error": f"Invalid capture type {parts['capture']}. must be 'confocal', 'fast', or 'long'"}
    # Validate extension
    if parts["ext"].lower()[1:] not in VALID_EXTENSIONS:
        return {"valid": False, "error": f"Invalid extension {parts['ext']}"}

    return {"valid": True, "parts": parts}

def is_numeric(value):
    """Return True if value is int or float (string-convertible)."""
    if value is None or value == "":
        return True  # allow empty values
    if isinstance(value, str) and value.strip().upper() in ALLOWED_NA:
        return True  # allow NA/N/A as "not applicable"
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False

def validate_numeric_fields(row: dict, file_category: str):
    """
    Validate numeric fields for a row based on file_category.
    Appends validation issues to `issues`.
    """
    issues = []
    expected_fields = NUMERIC_FIELDS.get(file_category, [])
    for field in expected_fields:
        if field in row and row[field] and not is_numeric(row[field]):
            issues.append(
                f"[{file_category}] Field '{field}' has non-numeric value: {row[field]}"
            )
    return issues

def validate_row(row, category):
    """
    Validate the data in a row. This function can be extended to include more complex validation logic.
    For now, it checks if required fields are present and if numeric fields can be converted to float.
    """
    issues = []
    # Check required fields
    missing = [f for f in REQUIRED_FIELDS.get(category, []) if not row.get(f)]
    if missing:
        issues.append(f"Missing required fields for {category}: {', '.join(missing)}")
    
    # Check if file_name is valid
    file_name = row.get("file_name", "")
    result = validate_filename(file_name)
    if not result["valid"]:
        issues.append(f"Filename error: {result['error']}")
    
    
    # Check data type and format
    # date should be YYYYMMDD
    date = row.get("date", "")
    if not re.match(r"\d{8}", str(row.get("date", ""))):
        issues.append(f"Invalid date format: {date}. it should be YYYYMMDD")
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        issues.append(f"Invalid date format: {date}. it should be YYYYMMDD")
    
    # replicate should be an integer
    try:
        int(row.get("replicate", ""))
    except ValueError:
        issues.append(f"Replicate of experiment is not integer: {row.get('replicate')}")
    
    # check the capture type
    capture_type = row.get("capture_type", "").lower()
    if capture_type not in CAPTURE_TYPES:
        issues.append(f"Invalid capture type: {row.get('capture_type')}. Must be one of {CAPTURE_TYPES}")
    
    # check the filed of view
    fov = row.get("field_of_view")
    try:
        int(fov)
    except (ValueError, TypeError):
        issues.append(f"Invalid field_of_view: {fov}. Must be an integer")

    # check the file extension
    if row["extension"] not in VALID_EXTENSIONS:
        issues.append(f"Invalid extension: {row.get('extension')}. Must be one of {VALID_EXTENSIONS}")
        
    # Check categorical fields
    if row.get("organism", "").lower() not in ALLOWED_ORGANISMS:
        issues.append(f"Invalid organism: {row.get('organism')}. Must be one of {ALLOWED_ORGANISMS}")

    
    

    issues.extend(validate_numeric_fields(row, category))

        
    # check the file path, decide later 


    '''
    # Check numeric fields
    numeric_fields = ["exposure_time", "time_interval", "dye_concentration_value", "laser_wavelength", "laser_intensity", "camera_binning", "pixel_size"]
    for field in numeric_fields:
        if field in row and row[field]:
            try:
                float(row[field])
            except ValueError:
                print(f"Invalid numeric value for {field}: {row[field]} in row: {row}")
                return False
    '''
    # ------------- raw file specific validation -------------
    if category == "raw":
        # check the laser intensity
        if row["laser_intensity"] and not re.match(r"^\d+%$", row.get("laser_intensity", "")):
            issues.append(f"Invalid laser intensity: {row['laser_intensity']}.  Must be in format 'N%' (e.g., '5%')")
        # check the dye concentration units
        if row["dye_concentration_unit"] and row.get("dye_concentration_unit", "") not in DYE_CONCENTRATION_UNITS and row.get("dye_concentration_unit", "").upper() not in ALLOWED_NA:
            issues.append(f"Invalid dye_concentration_unit: {row['dye_concentration_unit']}. Must be one of {DYE_CONCENTRATION_UNITS}")
        # check the condition concentration unit
        unit = row.get("concentration_unit", "")
        if row["concentration_unit"] and unit not in CONDITION_UNITS and unit.upper() not in ALLOWED_NA:
            issues.append(f"Invalid concentration_unit: {row['concentration_unit']}. Must be one of {CONDITION_UNITS}")

        # check the objective magnification
        if row["objective_magnification"] and not re.match(r"^\d+x$", row.get("objective_magnification", "")):
            issues.append(f"Invalid objective magnification: {row['objective_magnification']}. Must be in format 'Nx' (e.g., '100x')")
        # check the is_valid field
        if row["is_valid"] and row["is_valid"] not in ["Y", "N"]:
            issues.append(f"Invalid is_valid flag: {row['is_valid']}. Must be 'N' or 'Y")
        
        # check the user email format
        user_email = row.get("user_email")
        if user_email and not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", user_email):
            issues.append(f"Invalid email: {row['user_email']}")
        
        

    # -------------- mask specific validation -------------
    if category == "mask":
        # check the mask type
        if row["mask_type"] and row.get("mask_type", "").lower() not in MASK_TYPES:
            issues.append(f"Invalid mask_type: {row['mask_type']}")
        
        # decide later for validation of segmentation method and parameters



    # ------------ logging ----------
    if issues:
        logging.warning(f"Row {file_name} invalid: {issues}")
        return False, issues
    else:
        logging.info(f"Row {file_name} validated OK")
        return True, []
    
def validate_csv(file_path: str, output_path: str):
    invalid_rows = []
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            valid, issues = validate_row(row, row["file_category"].lower())
            if not valid:
                logging.warning(f"Row {i}: {issues}")
                invalid_rows.append({"row": row, "row_number": i, "issues": issues})

    # Save invalid rows to CSV
    if invalid_rows:
        fieldnames = list(invalid_rows[0]["row"].keys()) + ["row_number"] + ["issues"]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in invalid_rows:
                row_copy = item["row"].copy()
                row_copy["row_number"] = item["row_number"]
                row_copy["issues"] = "; ".join(item["issues"])
                writer.writerow(row_copy)

        logging.info(f"Saved {len(invalid_rows)} invalid rows to invalid_rows.csv, See validation.log for details.")
    else:
        logging.info("No invalid rows found.")

    return invalid_rows


# -------------- for the analysis metadata validation -----------
def validate_experiment_signature(signature: str):
    """
    Validate the experiment signature string.
    Expected format: YYYYMMDD|NN|organism|protein|condition|capture_type
    Returns a list of issues (empty if valid).
    """
    signature_issues = []
    
    parts = signature.split("|")
    if len(parts) != 6:
        return [f"Invalid format: expected 6 parts, got {len(parts)}"]
    
    date_str, replicate, organism, protein, condition, capture_type = parts
    
    # Validate date
    if not re.fullmatch(r"\d{8}", date_str):
        signature_issues.append(f"Invalid date format: {date_str}")
    else:
        try:
            datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            signature_issues.append(f"Invalid calendar date: {date_str}")
    
    # Validate replicate
    if not re.fullmatch(r"\d{2}", replicate):
        signature_issues.append(f"Invalid replicate format: {replicate}")

    # Validate organism
    if organism.lower() not in ALLOWED_ORGANISMS:
        signature_issues.append(f"Invalid organism: {organism}. Allowed: {ALLOWED_ORGANISMS}")
    # Validate protein
    if not re.fullmatch(r"[A-Za-z0-9]+", protein):
        signature_issues.append(f"Invalid protein format: {protein}. Must be alphanumeric")
    # Validate condition
    if not re.fullmatch(r"[A-Za-z0-9]+", condition):
        signature_issues.append(f"Invalid condition format: {condition}. Must be alphanumeric")
    
    # Validate capture type
    if capture_type.lower() not in CAPTURE_TYPES:
        signature_issues.append(f"Invalid capture type (expected): {CAPTURE_TYPES}")
    
    return signature_issues

def validate_analysis_row(row):
    issues = []

    # validate linked_experiment_signatures
    signature_issues = validate_experiment_signature(row["linked_experiment_signatures"])

    if signature_issues:
        issues.extend(signature_issues)

    # result_type
    if row.get("result_type") and row["result_type"] not in VALID_RESULT_TYPES:
        issues.append(f"Invalid result_type: {row['result_type']}")
    
    # result_value
    if row.get("result_value"):
        try:
            float(row["result_value"])
        except ValueError:
            issues.append(f"result_value not numeric: {row['result_value']}")

    # sample_size
    if row.get("sample_size"):
        try:
            int(row["sample_size"])
        except ValueError:
            issues.append(f"sample_size not integer: {row['sample_size']}")
    # standard_error
    if row.get("standard_error"):
        try:
            float(row["standard_error"])
        except ValueError:
            issues.append(f"standard_error not numeric: {row['standard_error']}")
    # analysis_method
    if row.get("result_type") and not row.get("analysis_method"):
        issues.append("Missing analysis_method for given result_type")

    if row.get("analysis_method") and row["analysis_method"] not in VALID_ANALYSIS_METHODS:
        issues.append(f"Invalid analysis_method: {row['analysis_method']}")

    # ---------------------------------   
    # analysis_parameters --> decide later, maybe a JSON string

    # analysis_files_path
    if not row.get("analysis_files_path") or not os.path.exists(row["analysis_files_path"]):
        issues.append(f"analysis_files_path not found: {row.get('analysis_files_path')}")

    return issues


def validate_analysis_metadata(file_path: str, output_path: str):
    """
    Validate analysis metadata CSV file.
    """
    invalid_rows = []
    with open(file_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            issues = validate_analysis_row(row)
            if issues:
                logging.warning(f"row {i} is invalid | Issues: {issues}")
                invalid_rows.append({"row": row, "row_number": i, "issues": issues})
            else:
                logging.info(f"row {i} is valid")

    # Save invalid rows if any
    if invalid_rows:
        with open(output_path, "w", newline="") as f:
            fieldnames = list(invalid_rows[0]["row"].keys()) + ["row_number"] + ["issues"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in invalid_rows:
                row_copy = item["row"].copy()
                row_copy["row_number"] = item["row_number"]
                row_copy["issues"] = "; ".join(item["issues"])
                writer.writerow(row_copy)

        logging.info(f"Saved {len(invalid_rows)} invalid rows to {output_path}, see validation.log for details.")
    else:
        logging.info("All rows valid")
    return invalid_rows

# Example usage
if __name__ == "__main__":
    metadata_csv_path = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/confocal/metadata_complete.csv"
    metadata_output_path = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/confocal/invalid_rows.csv"
    analysis_csv_path = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/dwell time/CPT/500ms interval/analysis_metadata2.csv"
    analysis_output_path = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/dwell time/CPT/500ms interval/invalid_analysis_rows.csv"
    invalids_rows = validate_csv(metadata_csv_path, metadata_output_path)
    validate_analysis_file = False
    if validate_analysis_file:
        invalid_analysis_rows = validate_analysis_metadata(analysis_csv_path, analysis_output_path)
    