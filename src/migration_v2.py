

import sqlite3

DB_PATH = "Reyes_lab_data.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# -------------------------------
# Drop all existing tables
# -------------------------------
tables = [
    "RawFiles", "TrackingFiles", "Masks", "AnalysisFiles",
    "AnalysisResults", "ExperimentAnalysisFiles", "AnalysisResultExperiments",
    "Experiment", "CaptureSetting", "StrainOrCellLine",
    "Protein", "Condition", "Organism", "User"
]

for table in tables:
    c.execute(f"DROP TABLE IF EXISTS {table};")

# -------------------------------
# Recreate all tables with proper constraints
# -------------------------------

# --- User ---
c.executescript("""
CREATE TABLE User (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT COLLATE NOCASE NOT NULL,
    last_name TEXT COLLATE NOCASE,
    email TEXT COLLATE NOCASE UNIQUE,
    UNIQUE(name, last_name)
);
""")

# --- Organism ---
c.executescript("""
CREATE TABLE Organism (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT COLLATE NOCASE NOT NULL
);
""")

# --- Protein ---
c.executescript("""
CREATE TABLE Protein (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT COLLATE NOCASE NOT NULL
);
""")

# --- StrainOrCellLine ---
c.executescript("""
CREATE TABLE StrainOrCellLine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT COLLATE NOCASE UNIQUE NOT NULL
);
""")

# --- Condition ---
c.executescript("""
CREATE TABLE Condition (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT COLLATE NOCASE NOT NULL,
    concentration_value FLOAT,
    concentration_unit TEXT COLLATE NOCASE,
    UNIQUE(name, concentration_value, concentration_unit)
);
""")

# --- CaptureSetting ---
c.executescript("""
CREATE TABLE CaptureSetting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_type TEXT COLLATE NOCASE NOT NULL,
    exposure_time FLOAT,
    time_interval FLOAT,
    fluorescent_dye TEXT COLLATE NOCASE,
    dye_concentration_value FLOAT,
    dye_concentration_unit TEXT COLLATE NOCASE,
    laser_wavelength FLOAT,
    laser_intensity FLOAT,
    camera_binning INTEGER,
    objective_magnification FLOAT,
    pixel_size FLOAT,
    UNIQUE(capture_type, exposure_time, time_interval, fluorescent_dye,
           dye_concentration_value, dye_concentration_unit, laser_wavelength,
           laser_intensity, camera_binning, objective_magnification, pixel_size)
);
""")

# --- Experiment ---
c.executescript("""
CREATE TABLE Experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organism_id INTEGER,
    protein_id INTEGER,
    strain_id INTEGER,
    condition_id INTEGER,
    capture_setting_id INTEGER,
    user_id INTEGER,
    date TEXT,
    replicate INTEGER,
    is_valid TEXT,
    comment TEXT,
    experiment_path TEXT,
    FOREIGN KEY (organism_id) REFERENCES Organism(id),
    FOREIGN KEY (protein_id) REFERENCES Protein(id),
    FOREIGN KEY (strain_id) REFERENCES StrainOrCellLine(id),
    FOREIGN KEY (condition_id) REFERENCES Condition(id),
    FOREIGN KEY (capture_setting_id) REFERENCES CaptureSetting(id),
    FOREIGN KEY (user_id) REFERENCES User(id),
    UNIQUE(organism_id, protein_id, strain_id, condition_id, capture_setting_id, user_id, date, replicate)
);
""")

# --- RawFiles ---
c.executescript("""
CREATE TABLE RawFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    file_name TEXT COLLATE NOCASE NOT NULL,
    field_of_view TEXT COLLATE NOCASE,
    file_type TEXT COLLATE NOCASE,
    file_path TEXT COLLATE NOCASE,
    UNIQUE(experiment_id, file_name, field_of_view, file_type),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);
""")

# --- TrackingFiles ---
c.executescript("""
CREATE TABLE TrackingFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    file_name TEXT COLLATE NOCASE NOT NULL,
    field_of_view TEXT COLLATE NOCASE,
    file_type TEXT COLLATE NOCASE,
    file_path TEXT COLLATE NOCASE,
    threshold FLOAT,
    linking_distance FLOAT,
    gap_closing_distance FLOAT,
    max_frame_gap INTEGER,
    UNIQUE(experiment_id, file_name, field_of_view, file_type, threshold, linking_distance, gap_closing_distance, max_frame_gap),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);
""")

# --- Masks ---
c.executescript("""
CREATE TABLE Masks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    mask_name TEXT COLLATE NOCASE NOT NULL,
    field_of_view TEXT COLLATE NOCASE,
    mask_type TEXT COLLATE NOCASE,
    file_type TEXT COLLATE NOCASE,
    mask_path TEXT COLLATE NOCASE,
    segmentation_method TEXT COLLATE NOCASE,
    segmentation_parameters TEXT COLLATE NOCASE,
    UNIQUE(experiment_id, mask_name, field_of_view, mask_type, file_type, segmentation_method, segmentation_parameters),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);
""")

# --- AnalysisFiles ---
c.executescript("""
CREATE TABLE AnalysisFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT COLLATE NOCASE NOT NULL,
    file_path TEXT COLLATE NOCASE,
    file_type TEXT COLLATE NOCASE,
    field_of_view TEXT COLLATE NOCASE,
    UNIQUE(file_name, file_path, file_type, field_of_view)
);
""")

# --- AnalysisResults ---
c.executescript("""
CREATE TABLE AnalysisResults (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_type TEXT,
    result_path TEXT,
    result_value FLOAT,
    sample_size INTEGER,
    standard_error FLOAT,
    Analysis_method TEXT,
    analysis_parameters TEXT,
    Analysis_file_path TEXT,
    UNIQUE(Analysis_file_path, result_type, result_value)
);
""")

# --- ExperimentAnalysisFiles ---
c.executescript("""
CREATE TABLE ExperimentAnalysisFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    analysis_file_id INTEGER NOT NULL,
    UNIQUE(experiment_id, analysis_file_id),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id),
    FOREIGN KEY (analysis_file_id) REFERENCES AnalysisFiles(id)
);
""")

# --- AnalysisResultExperiments ---
c.executescript("""
CREATE TABLE AnalysisResultExperiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_result_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    UNIQUE(analysis_result_id, experiment_id),
    FOREIGN KEY (analysis_result_id) REFERENCES AnalysisResults(id),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);
""")

conn.commit()
conn.close()
print("Database migration complete. All tables recreated with proper constraints.")
