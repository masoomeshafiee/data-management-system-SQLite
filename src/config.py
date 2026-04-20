

from dataclasses import dataclass, field
import os
import streamlit as st
from pathlib import Path

from typing import Dict, FrozenSet, List

import re


SHAREPOINT_SYNC_ROOT = (Path.home() / "McGill University" / "Reyes Lab_Group - Microscopy Data").resolve()

# ===============================
# General app configuration
# ===============================

@dataclass
class AppConfig:
    app_name: str = "Reyes Lab Data Manager"
    db_path: str = "app_database.db"
    debug_mode: bool = False

def load_config() -> AppConfig:
    app_name = os.environ.get("DB_APP_NAME", "Reyes Lab Data Manager")
    project_root = Path(__file__).resolve().parents[1]
    db_path = str((project_root / "database" / "Reyes_lab_data.db").resolve())
    debug_mode = os.environ.get("DB_APP_DEBUG_MODE", "False").lower() in ("true", "1", "t")

    return AppConfig(app_name=app_name, db_path=db_path, debug_mode=debug_mode)

# =========================================================
# Data intake/ validation Config / constants
# =========================================================

SUPPORTED_EXTS = {".csv", ".tif", ".tiff",".TIF", ".nd", ".npy", ".png", ".txt", ".json", ".mat", ".pickle", ".svg", ".pdf", ".xlsx"}
CAPTURE_TYPES = {"confocal", "fast", "long"}
ALLOWED_ORGANISMS = {"human", "yeast", "E.coli"}
MASK_TYPES = {"cell", "nucleus", "nucleus-g1", "membrane", "cytoplasm"}
CONDITION_UNITS = {"N/A","nM", "uM", "mM", "M", "%", "mJ/m2", "mJ/cm2", "J/cm2", "J/m2"}
ROLE_OPTIONS = ["unassigned", "raw", "mask", "tracking", "analysis", "batch_analysis", "plot", "config", "ignore"]


#DYE_CONCENTRATION_UNITS = {"pM","nM", "uM", "mM", "M", "N/A"} it is not used anymore to control the unit.

# Optional folder-name hints (NOT required)
FOLDER_HINTS = {
    "mask": "mask",
    "masks": "mask",
    "seg": "mask",
    "segmentation": "mask",
    "track": "tracking",
    "tracks": "tracking",
    "spot": "tracking",
    "spots": "tracking",
    "analysis": "analysis",
    "results": "analysis",
    "raw": "raw",
    "video": "raw",
    "images": "raw",
    "projection": "raw"
}



EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


REQUIRED_FIELDS = {
    "raw": ["file_name", "file_type", "date", "replicate", "organism", "protein", "strain", "condition", "capture_type"],
    "tracking": ["file_name", "file_type", "threshold"],
    "mask": ["file_name", "mask_type", "file_type", "segmentation_method"],
    "analysis_file": ["file_name", "file_type"]
}


# numeric fields per category
NUMERIC_FIELDS = {
    "raw": ["concentration_value", "exposure_time", "time_interval", "dye_concentration_value", "camera_binning", "pixel_size"],
    "tracking": ["threshold", "gap_closing_distance", "linking_distance", "max_frame_gap"],
    "mask": [],
    "analysis_file": []
}



# =========================================================
# Field registry for the search/browse page
# =========================================================

@dataclass(frozen=True)
class FieldDef:
    alias: str
    table: str
    column: str
    filterable: bool = True
    selectable: bool = True
    label: str | None = None
    data_type: str = "text"   # text, int, float, bool, date
    section: str = "general"
    default_visible: bool = False
    applies_to: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def sql(self) -> str:
        return f"{self.table}.{self.column}"

    @property
    def output_label(self) -> str:
        return self.label or self.alias


FIELD_REGISTRY: Dict[str, FieldDef] = {
    # reference tables:
    "organism_id": FieldDef("organism_id", "Organism", "id", data_type="int", section="sample", applies_to=frozenset({ "Organism"})),
    "protein_id": FieldDef("protein_id", "Protein", "id", data_type="int", section="sample", applies_to=frozenset({"Protein"})),
    "strain_id": FieldDef("strain_id", "StrainOrCellLine", "id", data_type="int", section="sample", applies_to=frozenset({"StrainOrCellLine"})),
    "condition_id": FieldDef("condition_id", "Condition", "id", data_type="int", section="sample", applies_to=frozenset({"Condition"})),
    "capture_setting_id": FieldDef("capture_setting_id", "CaptureSetting", "id", data_type="int", section="microscopy", applies_to=frozenset({"CaptureSetting"})),
    

    # Experiment
    "experiment_id": FieldDef("experiment_id", "Experiment", "id", data_type="int", section="experiment",
        default_visible=True, applies_to=frozenset({"Experiment"})),
    "date": FieldDef("date", "Experiment", "date", data_type="date", section="experiment",
        default_visible=True, applies_to=frozenset({"Experiment"})),
    "replicate": FieldDef("replicate", "Experiment", "replicate", data_type="int", section="experiment",
        default_visible=True, applies_to=frozenset({"Experiment"})),
    "is_valid": FieldDef("is_valid", "Experiment", "is_valid", data_type="text", section="experiment", default_visible=True, applies_to=frozenset({"Experiment"})),
    "comment": FieldDef("comment", "Experiment", "comment", data_type="text", section="experiment", applies_to=frozenset({"Experiment"})),
    "experiment_path": FieldDef("experiment_path", "Experiment", "experiment_path", data_type="text", section="experiment", default_visible=True, applies_to=frozenset({"Experiment"})),

    # Lookup entities (sample)
    "organism": FieldDef("organism", "Organism", "organism_name", section="sample", default_visible=True, applies_to=frozenset({"Experiment", "Organism"})),
    "protein": FieldDef("protein", "Protein", "protein_name", section="sample", default_visible=True, applies_to=frozenset({"Experiment", "Protein"})),
    "strain": FieldDef("strain", "StrainOrCellLine", "strain_name", section="sample", default_visible=True, applies_to=frozenset({"Experiment", "StrainOrCellLine"})),
    "condition": FieldDef("condition", "Condition", "condition_name", section="sample", default_visible=True, applies_to=frozenset({"Experiment", "Condition"})),
    "concentration_value": FieldDef("concentration_value", "Condition", "concentration_value", data_type="float", section="sample", default_visible=True, applies_to=frozenset({"Experiment", "Condition"})),
    "concentration_unit": FieldDef("concentration_unit", "Condition", "concentration_unit", section="sample", default_visible=True, applies_to=frozenset({"Experiment", "Condition"})),

    # Capture settings (microscopy)
    "capture_type": FieldDef("capture_type", "CaptureSetting", "capture_type", section="microscopy", default_visible=True, applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "exposure_time": FieldDef("exposure_time", "CaptureSetting", "exposure_time", data_type="float", section="microscopy", default_visible=True, applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "time_interval": FieldDef("time_interval", "CaptureSetting", "time_interval", data_type="float", section="microscopy", default_visible=True, applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "fluorescent_dye": FieldDef("fluorescent_dye", "CaptureSetting", "fluorescent_dye", section="microscopy", applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "dye_concentration_value": FieldDef("dye_concentration_value", "CaptureSetting", "dye_concentration_value", data_type="float", section="microscopy", applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "laser_wavelength": FieldDef("laser_wavelength", "CaptureSetting", "laser_wavelength", data_type="float", section="microscopy", applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "laser_intensity": FieldDef("laser_intensity", "CaptureSetting", "laser_intensity", data_type="float", section="microscopy", applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "camera_binning": FieldDef("camera_binning", "CaptureSetting", "camera_binning", data_type="int", section="microscopy", default_visible=True, applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "objective_magnification": FieldDef("objective_magnification", "CaptureSetting", "objective_magnification", data_type="float", section="microscopy", applies_to=frozenset({"Experiment", "CaptureSetting"})),
    "pixel_size": FieldDef("pixel_size", "CaptureSetting", "pixel_size", data_type="float", section="microscopy", applies_to=frozenset({"Experiment", "CaptureSetting"})),

    # User
    "user_id": FieldDef("user_id", "User", "id", data_type="int", section="user", applies_to=frozenset({"Experiment", "User"})),
    "user_name": FieldDef("user_name", "User", "user_name", section="user", default_visible=True, applies_to=frozenset({"Experiment", "User"})),
    "last_name": FieldDef("last_name", "User", "last_name", section="user", default_visible=True, applies_to=frozenset({"Experiment", "User"})),
    "email": FieldDef("email", "User", "email", section="user", applies_to=frozenset({"Experiment", "User"})),

    # RawFiles
    "raw_file_id": FieldDef("raw_file_id", "RawFiles", "id", data_type="int",section="raw_files", applies_to=frozenset({"RawFiles"})),
    "raw_file_name": FieldDef("raw_file_name", "RawFiles", "file_name", section="raw_files", default_visible=True, applies_to=frozenset({"RawFiles"})),
    "raw_file_type": FieldDef("raw_file_type", "RawFiles", "file_type", section="raw_files", default_visible=True, applies_to=frozenset({"RawFiles"})),
    "raw_file_path": FieldDef("raw_file_path", "RawFiles", "file_path", section="raw_files", default_visible=True, applies_to=frozenset({"RawFiles"})),

    # TrackingFiles
    "tracking_file_id": FieldDef("tracking_file_id", "TrackingFiles", "id", data_type="int", section="tracking_files", applies_to=frozenset({"TrackingFiles"})),
    "tracking_file_name": FieldDef("tracking_file_name", "TrackingFiles", "file_name", section="tracking_files", default_visible=True, applies_to=frozenset({"TrackingFiles"})),
    "tracking_file_type": FieldDef("tracking_file_type", "TrackingFiles", "file_type", section="tracking_files", default_visible=True, applies_to=frozenset({"TrackingFiles"})),
    "tracking_file_path": FieldDef("tracking_file_path", "TrackingFiles", "file_path", section="tracking_files", default_visible=True, applies_to=frozenset({"TrackingFiles"})),
    "threshold": FieldDef("threshold", "TrackingFiles", "threshold", data_type="float", section="tracking_files", applies_to=frozenset({"TrackingFiles"})),
    "linking_distance": FieldDef("linking_distance", "TrackingFiles", "linking_distance", data_type="float", section="tracking_files", applies_to=frozenset({"TrackingFiles"})),
    "gap_closing_distance": FieldDef("gap_closing_distance", "TrackingFiles", "gap_closing_distance", data_type="float", section="tracking_files", applies_to=frozenset({"TrackingFiles"})),
    "max_frame_gap": FieldDef("max_frame_gap", "TrackingFiles", "max_frame_gap", data_type="int", section="tracking_files", applies_to=frozenset({"TrackingFiles"})),
    "trackmate_settings_json": FieldDef("trackmate_settings_json", "TrackingFiles", "trackmate_settings_json", section="tracking_files", applies_to=frozenset({"TrackingFiles"})),

    # Masks
    "mask_id": FieldDef("mask_id", "Masks", "id", data_type="int", section="masks", applies_to=frozenset({"Masks"})),
    "mask_name": FieldDef("mask_name", "Masks", "file_name", section="masks", default_visible=True, applies_to=frozenset({"Masks"})),
    "mask_type": FieldDef("mask_type", "Masks", "mask_type", section="masks", default_visible=True, applies_to=frozenset({"Masks"})),
    "mask_file_type": FieldDef("mask_file_type", "Masks", "file_type", section="masks", default_visible=True, applies_to=frozenset({"Masks"})),
    "mask_path": FieldDef("mask_path", "Masks", "file_path", section="masks", default_visible=True, applies_to=frozenset({"Masks"})),
    "segmentation_method": FieldDef("segmentation_method", "Masks", "segmentation_method", section="masks", applies_to=frozenset({"Masks"})),
    "segmentation_parameters": FieldDef("segmentation_parameters", "Masks", "segmentation_parameters", section="masks", applies_to=frozenset({"Masks"})),

    # AnalysisFiles
    "analysis_file_id": FieldDef("analysis_file_id", "AnalysisFiles", "id", data_type="int", section="analysis_files", applies_to=frozenset({"AnalysisFiles"})),
    "analysis_file_name": FieldDef("analysis_file_name", "AnalysisFiles", "file_name", section="analysis_files", default_visible=True, applies_to=frozenset({"AnalysisFiles"})),
    "analysis_file_type": FieldDef("analysis_file_type", "AnalysisFiles", "file_type", section="analysis_files", default_visible=True, applies_to=frozenset({"AnalysisFiles"})),
    "analysis_file_path": FieldDef("analysis_file_path", "AnalysisFiles", "file_path", section="analysis_files", default_visible=True, applies_to=frozenset({"AnalysisFiles"})),

    # Results
    "analysis_result_id": FieldDef("analysis_result_id", "Results", "id", data_type="int", section="results", applies_to=frozenset({"Results"})),
    "analysis_result_type": FieldDef("analysis_result_type", "Results", "result_type", section="results", default_visible=True, applies_to=frozenset({"Results"})),
    "result_value": FieldDef("result_value", "Results", "result_value", data_type="float", section="results", default_visible=True, applies_to=frozenset({"Results"})),
    "sample_size": FieldDef("sample_size", "Results", "sample_size", data_type="int", section="results", default_visible=True, applies_to=frozenset({"Results"})),
    "standard_error": FieldDef("standard_error", "Results", "standard_error", data_type="float", section="results", default_visible=True, applies_to=frozenset({"Results"})),
    "analysis_method": FieldDef("analysis_method", "Results", "analysis_method", section="results", applies_to=frozenset({"Results"})),
    "analysis_parameters_json": FieldDef("analysis_parameters", "Results", "analysis_parameters_json", section="results", applies_to=frozenset({"Results"})),
}

# =========================================================
# Relationships / join graph
# =========================================================

TABLE_RELATIONSHIPS = {
    "Experiment": {
        "organism_id": "Organism",
        "protein_id": "Protein",
        "strain_id": "StrainOrCellLine",
        "condition_id": "Condition",
        "user_id": "User",
        "capture_setting_id": "CaptureSetting",
    },
    "RawFiles": {"experiment_id": "Experiment"},
    "TrackingFiles": {"experiment_id": "Experiment"},
    "Masks": {"experiment_id": "Experiment"},
    "Experiment_Analysis_Files_Link": {
        "experiment_id": "Experiment",
        "analysis_file_id": "AnalysisFiles",
    },
    "Results_Analysis_Files_Link": {
        "result_id": "Results",
        "analysis_file_id": "AnalysisFiles",
    },
}


REFERENCE_PARENT_TABLES = {
    "Organism",
    "Protein",
    "StrainOrCellLine",
    "Condition",
    "CaptureSetting",
    "User",
}


REFERENCE_PARENT_CONFIG = {
    parent_table: fk_col
    for fk_col, parent_table in TABLE_RELATIONSHIPS["Experiment"].items()
}


# =======================================================
# Experiment browsing / search config
# =======================================================
BASE_EXPERIMENT_TABLES = {
    "Experiment",
    "Organism",
    "Protein",
    "StrainOrCellLine",
    "Condition",
    "CaptureSetting",
    "User",
}


TARGET_OPTIONS = [
        "Experiment",
        "Condition",
        "Organism",
        "Protein",
        "StrainOrCellLine",
        "CaptureSetting",
        "User",
        "RawFiles",
        "TrackingFiles",
        "Masks",
        "AnalysisFiles",
        "Results",
    ]

BASE_EXPERIMENT_QUERY = """
SELECT
    Experiment.id AS experiment_id,
    Organism.organism_name AS organism,
    Protein.protein_name AS protein,
    StrainOrCellLine.strain_name AS strain,
    Condition.condition_name AS condition,
    Condition.concentration_value AS concentration_value,
    Condition.concentration_unit AS concentration_unit,
    CaptureSetting.capture_type AS capture_type,
    CaptureSetting.exposure_time AS exposure_time,
    CaptureSetting.time_interval AS time_interval,
    User.user_name AS user_name,
    User.email AS email,
    Experiment.date AS date,
    Experiment.replicate AS replicate,
    Experiment.is_valid AS is_valid,
    Experiment.comment AS comment,
    Experiment.experiment_path AS experiment_path
FROM Experiment
JOIN Organism ON Experiment.organism_id = Organism.id
JOIN Protein ON Experiment.protein_id = Protein.id
JOIN StrainOrCellLine ON Experiment.strain_id = StrainOrCellLine.id
JOIN Condition ON Experiment.condition_id = Condition.id
JOIN CaptureSetting ON Experiment.capture_setting_id = CaptureSetting.id
LEFT JOIN User ON Experiment.user_id = User.id
"""


