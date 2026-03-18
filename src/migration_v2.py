
import os
import sqlite3

print(sqlite3.sqlite_version)
DB_PATH = os.environ.get("DB_PATH", "../db/Reyes_lab_data.db")
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
'''
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

'''
# ------------------------- Schema updates for v2.0 -------------------------
# needed to be compatible with new version of Streamlit app and data validation and db operations
# ----------------------------------------------------------------------------

'''
c.execute("""ALTER TABLE TrackingFiles ADD COLUMN trackmate_settings_json TEXT;""")

c.execute("""ALTER TABLE TrackingFiles DROP COLUMN field_of_view;""")

c.execute("""ALTER TABLE RawFiles DROP COLUMN field_of_view;""")

c.execute("""ALTER TABLE Masks DROP COLUMN field_of_view;""")
c.execute("""ALTER TABLE Masks RENAME COLUMN mask_name TO file_name;""")
c.execute("""ALTER TABLE Masks RENAME COLUMN mask_path TO file_path;""")

c.execute("""ALTER TABLE AnalysisFiles Drop COLUMN field_of_view;""")



# 1) Create new table without field_of_view
c.executescript("""
CREATE TABLE TrackingFiles_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    file_name TEXT COLLATE NOCASE NOT NULL,
    file_type TEXT COLLATE NOCASE,
    file_path TEXT COLLATE NOCASE,
    threshold FLOAT,
    linking_distance FLOAT,
    gap_closing_distance FLOAT,
    max_frame_gap INTEGER,
    trackmate_settings_json TEXT,
    UNIQUE(experiment_id, file_name, file_type, threshold, linking_distance, gap_closing_distance, max_frame_gap),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);
""")

# 2) Copy data over (exclude field_of_view)
c.executescript("""
INSERT INTO TrackingFiles_new (
    id, experiment_id, file_name, file_type, file_path,
    threshold, linking_distance, gap_closing_distance, max_frame_gap,
    trackmate_settings_json
)
SELECT
    id, experiment_id, file_name, file_type, file_path,
    threshold, linking_distance, gap_closing_distance, max_frame_gap,
    trackmate_settings_json
FROM TrackingFiles;
""")

# 3) Swap tables
c.executescript("""
DROP TABLE TrackingFiles;
ALTER TABLE TrackingFiles_new RENAME TO TrackingFiles;
""")
'''



# --------------------------------
# Create the tables. 
# --------------------------------

# tables = [
#     "RawFiles", "TrackingFiles", "Masks", "AnalysisFiles",
#     "AnalysisResults", "ExperimentAnalysisFiles", "AnalysisResultExperiments",
#     "Experiment", "CaptureSetting", "StrainOrCellLine",
#     "Protein", "Condition", "Organism", "User"
# ]

# for table in tables:
#     c.execute(f"DROP TABLE IF EXISTS {table};")

# c.executescript("""
#     PRAGMA foreign_keys = ON;
                            
# CREATE TABLE IF NOT EXISTS Organism (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         organism_name TEXT NOT NULL UNIQUE COLLATE NOCASE
#                 CHECK (trim(organism_name) <> '')
#     );
# CREATE TABLE StrainOrCellLine (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     strain_name TEXT COLLATE NOCASE UNIQUE NOT NULL
#     CHECK (trim(strain_name) <> '')
# );
# CREATE TABLE Protein (
#     id INTEGER PRIMARY KEY AUTOINCREMENT,
#     protein_name TEXT COLLATE NOCASE NOT NULL
#                 CHECK (trim(protein_name) <> '')
# );
                

# CREATE TABLE Condition (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         condition_name TEXT NOT NULL COLLATE NOCASE CHECK (trim(condition_name) <> ''),
#         concentration_value FLOAT CHECK (concentration_value >= 0),
#         concentration_unit TEXT COLLATE NOCASE 
#     );
# CREATE TABLE CaptureSetting (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         capture_type TEXT NOT NULL COLLATE NOCASE CHECK (trim(capture_type) <> ''),
#         exposure_time FLOAT NOT NULL CHECK (exposure_time > 0),
#         time_interval FLOAT NOT NULL CHECK (time_interval > 0),
#         fluorescent_dye TEXT COLLATE NOCASE NOT NULL CHECK (trim(fluorescent_dye) <> ''),
#         dye_concentration_value FLOAT NOT NULL CHECK (dye_concentration_value > 0),
#         dye_concentration_unit TEXT COLLATE NOCASE CHECK (dye_concentration_unit IS NULL OR trim(dye_concentration_unit) <> ''),
#         laser_wavelength FLOAT CHECK (laser_wavelength > 0),
#         laser_intensity FLOAT NOT NULL CHECK (laser_intensity > 0),
#         camera_binning INTEGER CHECK (camera_binning IS NULL OR camera_binning > 0),
#         objective_magnification FLOAT CHECK (objective_magnification IS NULL or objective_magnification > 0),
#         pixel_size FLOAT CHECK (pixel_size IS NULL or pixel_size > 0)

#     );
# CREATE TABLE User (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         user_name TEXT NOT NULL COLLATE NOCASE CHECK (trim(user_name) <> ''),
#         last_name TEXT COLLATE NOCASE CHECK (trim(last_name) <> ''),
#         email TEXT UNIQUE COLLATE NOCASE CHECK (trim(email) <> '')
#     );

# CREATE TABLE Experiment (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         organism_id INTEGER NOT NULL,
#         protein_id INTEGER NOT NULL,
#         strain_id INTEGER NOT NULL,
#         condition_id INTEGER NOT NULL,
#         capture_setting_id INTEGER NOT NULL,
#         user_id INTEGER NOT NULL,
#         date TEXT NOT NULL CHECK (trim(date) <> ''),
#         replicate INTEGER NOT NULL CHECK (replicate >= 1),
#         is_valid BOOLEAN NOT NULL CHECK (is_valid IN (0, 1)),
#         comment TEXT,
#         experiment_path TEXT NOT NULL COLLATE NOCASE,
#         FOREIGN KEY (organism_id) REFERENCES Organism(id),
#         FOREIGN KEY (protein_id) REFERENCES Protein(id),
#         FOREIGN KEY (strain_id) REFERENCES StrainOrCellLine(id),
#         FOREIGN KEY (condition_id) REFERENCES Condition(id),
#         FOREIGN KEY (capture_setting_id) REFERENCES CaptureSetting(id),
#         FOREIGN KEY (user_id) REFERENCES User(id)
#     );
# CREATE TABLE RawFiles (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         experiment_id INTEGER NOT NULL,
#         file_name TEXT NOT NULL COLLATE NOCASE CHECK (trim(file_name) <> ''),
#         file_type TEXT COLLATE NOCASE,
#         file_path TEXT NOT NULL COLLATE NOCASE CHECK (trim(file_path) <> ''),
#         FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
#     );
# CREATE TABLE TrackingFiles (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         experiment_id INTEGER NOT NULL,
#         file_name TEXT NOT NULL COLLATE NOCASE CHECK (trim(file_name) <> ''),
#         file_type TEXT COLLATE NOCASE,
#         file_path TEXT NOT NULL COLLATE NOCASE CHECK (trim(file_path) <> ''),
#         threshold FLOAT CHECK (threshold > 0),
#         linking_distance FLOAT CHECK (linking_distance > 0),
#         gap_closing_distance FLOAT CHECK (gap_closing_distance > 0),
#         max_frame_gap INTEGER CHECK (max_frame_gap >= 0),
#         trackmate_settings_json TEXT,
#         FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
        
#     );


# CREATE TABLE Masks (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         experiment_id INTEGER NOT NULL,
#         mask_name TEXT NOT NULL COLLATE NOCASE CHECK (trim(mask_name) <> ''),
#         mask_type TEXT COLLATE NOCASE CHECK (trim(mask_type) <> ''),
#         file_type TEXT COLLATE NOCASE,
#         mask_path TEXT NOT NULL COLLATE NOCASE CHECK (trim(mask_path) <> ''),
#         segmentation_method TEXT COLLATE NOCASE CHECK (trim(segmentation_method) <> ''),
#         segmentation_parameters TEXT COLLATE NOCASE,
#         FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE
#     );
# CREATE TABLE AnalysisFiles (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         file_name TEXT NOT NULL COLLATE NOCASE CHECK (trim(file_name) <> ''),
#         file_path TEXT NOT NULL COLLATE NOCASE CHECK (trim(file_path) <> ''),
#         file_type TEXT COLLATE NOCASE
#     );

# CREATE TABLE Experiment_Analysis_Files_Link (
#         experiment_id INTEGER NOT NULL,
#         analysis_file_id INTEGER NOT NULL,
#         PRIMARY KEY(experiment_id, analysis_file_id),
#         FOREIGN KEY (experiment_id) REFERENCES Experiment(id) ON DELETE CASCADE,
#         FOREIGN KEY (analysis_file_id) REFERENCES AnalysisFiles(id) ON DELETE CASCADE
#     );
# CREATE TABLE Results (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         result_type TEXT COLLATE NOCASE NOT NULL CHECK (trim(result_type) <> ''),
#         result_value FLOAT NOT NULL,
#         sample_size INTEGER CHECK (sample_size IS NULL OR sample_size >= 1),
#         standard_error FLOAT CHECK (standard_error IS NULL OR standard_error >= 0),
#         analysis_method TEXT COLLATE NOCASE NOT NULL CHECK (trim(analysis_method) <> ''),
#         analysis_parameters_json TEXT
        
#     );
# CREATE TABLE Result_Analysis_Files_Link (
#         result_id INTEGER NOT NULL,
#         analysis_file_id INTEGER NOT NULL,
#         PRIMARY KEY(result_id, analysis_file_id),
#         FOREIGN KEY (result_id) REFERENCES Results(id) ON DELETE CASCADE, 
#         FOREIGN KEY (analysis_file_id) REFERENCES AnalysisFiles(id) ON DELETE CASCADE
#     );

# -- =========================
# -- Unique indexes (NULL-safe where needed)
# -- =========================
# CREATE UNIQUE INDEX condition_identity
# ON Condition (
#     condition_name,
#     COALESCE(concentration_value, -1.0e308),
#     COALESCE(concentration_unit, '')
# );

# CREATE UNIQUE INDEX capture_setting_identity
# ON CaptureSetting (
#     capture_type,
#     COALESCE(exposure_time, -1.0e308),
#     COALESCE(time_interval, -1.0e308),
#     COALESCE(fluorescent_dye, ''),
#     COALESCE(dye_concentration_value, -1.0e308),
#     COALESCE(laser_intensity, -1.0e308)
# );
# CREATE UNIQUE INDEX experiment_identity
#     ON Experiment (
#         COALESCE(organism_id, -1),
#         COALESCE(protein_id, -1),
#         COALESCE(strain_id, -1),
#         COALESCE(condition_id, -1),
#         COALESCE(capture_setting_id, -1),
#         COALESCE(date, ''),
#         COALESCE(replicate, -1)
#     );

# CREATE UNIQUE INDEX rawfiles_identity
# ON RawFiles (experiment_id, file_name, file_path);

# CREATE UNIQUE INDEX trackingfiles_identity
# ON TrackingFiles (experiment_id, file_name, file_path, threshold, linking_distance, gap_closing_distance, max_frame_gap);

# CREATE UNIQUE INDEX masks_identity
# ON Masks (experiment_id, mask_name, mask_path);

# CREATE UNIQUE INDEX analysisfiles_identity
# ON AnalysisFiles (file_name, file_path, file_type);
           
# CREATE UNIQUE INDEX results_identity
# ON Results (result_type, result_value, analysis_method);

            
    
# """)

#c.execute("ALTER TABLE CaptureSetting ADD COLUMN pixel_size_unit TEXT COLLATE NOCASE CHECK (trim(pixel_size_unit) <> '');")

c.execute("ALTER TABLE CaptureSetting DROP COLUMN dye_concentration_unit;")
conn.commit()
conn.close()
print("Migration to v2.0 completed successfully. All tables are recreated with proper constraints.")


