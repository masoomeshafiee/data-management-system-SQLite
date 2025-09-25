'''import os
import csv
import re
from pathlib import Path
from collections import defaultdict

ANALYSIS_EXTENSIONS = {".mat", ".svg", ".pdf", ".xlsx"}

# Parse file name for analysis (can reuse your existing parse_filename)
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

def is_analysis_file(path: Path) -> bool:
    """Check if file is a recognized analysis result."""
    return path.suffix.lower() in ANALYSIS_EXTENSIONS


def collect_analysis_metadata(base_dir, output_csv):
    analysis_records = []
    for root, _, files in os.walk(base_dir):
        # Skip if no analysis files in this folder
        analysis_files = [f for f in files if is_analysis_file(Path(f))]
        if not analysis_files:
            continue
        
        # Collect experiment signatures from any TrackMate CSV files here
        exp_signatures = set()
        for f in files:
            if f.lower().endswith((".csv",)) and ("spots" in f.lower() or "tracks" in f.lower()):
                parsed = parse_filename(f)
                if parsed:
                    signature = "|".join([
                        parsed["date"],
                        parsed["replicate"],
                        parsed["organism"],
                        parsed["protein"],
                        parsed["condition"],
                        parsed["capture_type"]
                    ])
                    exp_signatures.add(signature)
                # Only add a record if we actually found experiment links
        if exp_signatures:
            analysis_records.append({
                "linked_experiment_signatures": list(exp_signatures),
                "result_type": "",
                "result_value": "",
                "sample_size": "",
                "standard_error": "",
                "analysis_method": "",
                "analysis_parameters": "",
                "analysis_files_path": root
            })

    if analysis_records:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=analysis_records[0].keys())
            writer.writeheader()
            writer.writerows(analysis_records)
        print(f" Metadata for {len(analysis_records)} analysis saved to {output_csv}")
    else:
        print("No analysis files found.")

if __name__ == "__main__":
    base_dir = '/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/dwell time/CPT/500ms interval/20240122_Rfa1_long_549_500ms_5%_CPT_40uM_50nM/Analysis_COST_thr150.0_lnk3.0_gp5.0__cost_INT0.3'
    output_csv = os.path.join(base_dir,"analysis_metadata.csv")
    collect_analysis_metadata(base_dir, output_csv)
'''
import os
import csv
import re
from pathlib import Path

ANALYSIS_EXTENSIONS = {".mat", ".svg", ".pdf", ".xlsx"}  # add more if needed
CSV_KEYWORDS = ("spots", "tracks")  # keywords for experiment CSVs

def parse_filename(filename):
    """Parse filename into structured fields for experiment signature."""
    name, ext = os.path.splitext(filename)
    pattern = re.compile(
    r"(?P<date>\d{8})-(?P<replicate>\d+)_"
    r"(?P<organism>[^_]+)_(?P<protein>[^_]+)_(?P<condition>[^_]+)_"
    r"(?P<capture_type>[^_]+)_(?P<field_of_view>\d+)_(?P<file_type>.+)"
)
    match = pattern.match(name)
    if not match:
        return None
    return {**match.groupdict(), "extension": ext.lstrip(".")}

def is_analysis_file(path: Path) -> bool:
    """Check if file is a recognized analysis result."""
    return path.suffix.lower() in ANALYSIS_EXTENSIONS

def find_experiment_signatures(search_dir: Path):
    """Find experiment signatures from CSV or raw files in a directory."""
    exp_signatures = set()
    for f in os.listdir(search_dir):
        lower_name = f.lower()
        if lower_name.endswith(".csv") and any(k in lower_name for k in CSV_KEYWORDS):
            parsed = parse_filename(f)
            if parsed:
                signature = "|".join([
                    parsed["date"],
                    parsed["replicate"],
                    parsed["organism"],
                    parsed["protein"],
                    parsed["condition"],
                    parsed["capture_type"]
                ])
                exp_signatures.add(signature)
        else:
            # Try parsing raw files (same naming pattern but other extensions)
            parsed = parse_filename(f)
            if parsed:
                signature = "|".join([
                    parsed["date"],
                    parsed["replicate"],
                    parsed["organism"],
                    parsed["protein"],
                    parsed["condition"],
                    parsed["capture_type"]
                ])
                exp_signatures.add(signature)
    return exp_signatures

def collect_analysis_metadata(base_dir, output_csv):
    analysis_records = []
    for root, _, files in os.walk(base_dir):
        # Check for analysis files in current folder
        analysis_files = [f for f in files if is_analysis_file(Path(f))]
        if not analysis_files:
            continue

        analysis_path = Path(root)
        parent_dir = analysis_path.parent

        # Look in both parent directory and current analysis folder for CSV/raw files
        exp_signatures = set()
        exp_signatures |= find_experiment_signatures(parent_dir)
        exp_signatures |= find_experiment_signatures(analysis_path)

        if exp_signatures:
            analysis_records.append({
                "linked_experiment_signatures": ",".join(sorted(exp_signatures)),
                "result_type": "",
                "result_value": "",
                "sample_size": "",
                "standard_error": "",
                "analysis_method": "",
                "analysis_parameters": "",
                "analysis_files_path": str(analysis_path)
            })

    if analysis_records:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=analysis_records[0].keys())
            writer.writeheader()
            writer.writerows(analysis_records)
        print(f"Metadata for {len(analysis_records)} analysis saved to {output_csv}")
    else:
        print("No analysis files found.")

if __name__ == "__main__":
    base_dir = '/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/dwell time/CPT/500ms interval'
    output_csv = os.path.join(base_dir, "analysis_metadata2.csv")
    collect_analysis_metadata(base_dir, output_csv)
