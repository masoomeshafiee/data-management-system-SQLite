import os
import csv
import re
from pathlib import Path
from collections import defaultdict

# --- Parse structured file name ---
def parse_filename(filename):
    name, ext = os.path.splitext(filename)
    pattern = re.compile(
        r"(?P<date>\d{8})-(?P<replicate>\d+)_"
        r"(?P<organism>[^_]+)_(?P<protein>[^_]+)_(?P<condition>[^_]+)_"
        r"(?P<capture_type>[^_]+)_(?P<field_of_view>\d+)_(?P<file_type>.+)"
    )
    match = pattern.match(name)
    if not match:
        return None
    return {
        **match.groupdict(),
        "extension": ext.lstrip(".")
    }

# --- Detect file category ---
def detect_file_category(file_name, extension):
    name_lower = file_name.lower()
    ext_lower = extension.lower()
    if "mask" in name_lower or ext_lower in ("npy", "txt"):
        return "mask"
    elif ("track" in name_lower and ext_lower in ("csv",)) or ("spot" in name_lower and ext_lower in ("csv",)):
        return "tracking"
    elif "analysis" in name_lower or ext_lower in ("mat", "json", "pickle",".svg", ".pdf", ".xlsx"):
        return "analysis_file"
    else:
        return "raw"  # default fallback

# --- Main metadata generation ---
def generate_core_metadata(base_dir, output_csv):
    rows = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            parsed = parse_filename(f)
            if parsed:
                parsed["file_name"] = f
                #parsed["full_path"] = os.path.join(root, f)   # keep full path
                parsed["folder"] = os.path.basename(root)     # just the folder name
                file_category = detect_file_category(f, parsed["extension"])
                parsed["file_category"] = file_category

                rows.append(parsed)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Metadata file with blanks saved to: {output_csv}")
    else:
        print("No valid files found.")

# Fields per file type

FIELDS_BY_TYPE = {
    "raw": {
        "global": [
            "user_name", "user_last_name", "user_email",
            "fluorescent_dye", "dye_concentration_value", "dye_concentration_unit", "strain",
            "objective_magnification", "laser_wavelength", "laser_intensity",
            "camera_binning", "pixel_size"
        ],
        "per_experiment": [
            "exposure_time", "time_interval", "concentration_value",
            "comment", "is_valid", "file_path", "concentration_unit"
        ],
    },
    "mask": {
        "global": [
            "segmentation_method", "segmentation_parameters", "mask_type"
        ],
        "per_experiment": ["strain"],
    },
    "tracking": {
        "global": ["linking_distance", "gap_closing_distance", "max_frame_gap"],
        "per_experiment": [
            "threshold", "strain"
        ],
    },
}

def prompt_for_global_fields(rows):
    # Group rows by file_category only
    grouped = defaultdict(list)
    for row in rows:
        ftype = row.get("file_category", "raw").lower()
        grouped[ftype].append(row)

    for ftype, group_rows in grouped.items():
        global_fields = FIELDS_BY_TYPE.get(ftype, {}).get("global", [])
        if not global_fields:
            continue

        print(f"\n--- Filling global fields for file type: {ftype} ---")
        batch_values = {}
        for field in global_fields:
            val = input(f"Enter value for global field '{field}' (leave blank to skip): ")
            if val.strip():
                batch_values[field] = val.strip()

        for row in group_rows:
            for field in global_fields:
                if not row.get(field):
                    row[field] = batch_values.get(field, row.get(field, ""))


def prompt_for_experiment_fields(rows):
    grouped = defaultdict(list)
    for row in rows:
        ftype = row.get("file_category", "raw").lower()
        date = row.get("date", "unknown")
        folder = row.get("folder", "unknown")
        grouped[(ftype, date, folder)].append(row)

    for (ftype, date, folder), group_rows in grouped.items():
        per_exp_fields = FIELDS_BY_TYPE.get(ftype, {}).get("per_experiment", [])
        if not per_exp_fields:
            continue

        print(f"\n--- Filling per-experiment fields for file type: {ftype}, date {date}, folder name {folder} ---")
        batch_values = {}
        for field in per_exp_fields:
            val = input(f"Enter value for '{field}' (leave blank to skip, applies to all {len(group_rows)} files): ")
            if val.strip():
                batch_values[field] = val.strip()

        for row in group_rows:
            for field in per_exp_fields:
                if not row.get(field):
                    row[field] = batch_values.get(field, row.get(field, ""))

def complete_metadata(core_csv, output_csv):
    with open(core_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    prompt_for_global_fields(rows)         # once per file category
    prompt_for_experiment_fields(rows)     # grouped by (category, date)

    # Save
    # Your preferred order
    PREFERRED_ORDER = [
        "file_name", "date", "replicate", "capture_type", "field_of_view",
        "file_type", "extension", "organism", "strain", "protein", "condition",
        "condition_concentration", "concentration_value", "concentration_unit", "exposure_time", "time_interval",
        "laser_wavelength", "laser_intensity", "fluorescent_dye", "dye_concentration_value",
        "dye_concentration_unit", "objective_magnification", "camera_binning", "pixel_size",
        "is_valid", "comment", "user_name", "user_last_name", "user_email", "file_category",
        "mask_type", "segmentation_method", "segmentation_parameters",
        "threshold", "gap_closing_distance", "linking_distance", "max_frame_gap",
        "file_path"
    ]

    # Collect all fieldnames from the data
    all_fieldnames = set()
    for row in rows:
        all_fieldnames.update(row.keys())

    # Keep preferred order first, then add any new/unexpected fields at the end
    fieldnames = [col for col in PREFERRED_ORDER if col in all_fieldnames]
    extra_cols = [col for col in all_fieldnames if col not in fieldnames]
    fieldnames.extend(extra_cols)

    # Write CSV with the ordered columns
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetadata completion saved to: {output_csv}")


# --- Example usage ---
if __name__ == "__main__":
    #base_dir = "/Users/masoomeshafiee/Library/CloudStorage/OneDrive-SharedLibraries-McGillUniversity/Reyes Lab_Group - Documents/Microscopy Data/Masoumeh/test"  # UPDATE THIS
    base_dir = '/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/confocal'
    output_csv = os.path.join(base_dir, "metadata_core.csv")
    generate_core_metadata(base_dir, output_csv)


    
    complete_csv_path = os.path.join(base_dir, "metadata_complete.csv")
    complete_metadata(output_csv, complete_csv_path)