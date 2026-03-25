import sqlite3

conn = sqlite3.connect("Reyes_lab_data.db")
cursor = conn.cursor()

cursor.executescript("""
CREATE TABLE IF NOT EXISTS Organism (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Protein (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS StrainOrCellLine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS Condition (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    concentration_value Real UNIQUE,
    concentration_unit TEXT UNIQUE                       
);

CREATE TABLE IF NOT EXISTS User (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT
);

CREATE TABLE IF NOT EXISTS CaptureSetting (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    capture_type TEXT NOT NULL,  -- e.g., 'confocal', 'fast', 'long'
    exposure_time REAL,
    time_interval REAL,
    fluorescent_dye TEXT,
    dye_concentration_value REAL,
    dye_concentration_unit TEXT,
    laser_wavelength TEXT,
    laser_intensity TEXT,
    camera_binning INTEGER,
    objective_magnification TEXT,
    pixel_size REAL
);

CREATE TABLE IF NOT EXISTS Experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organism_id INTEGER,
    protein_id INTEGER,
    strain_id INTEGER,
    condition_id INTEGER,
    capture_setting_id INTEGER,
    user_id INTEGER,
    experiment_path TEXT NOT NULL,
    date TEXT,
    FOREIGN KEY (organism_id) REFERENCES Organism(id),
    FOREIGN KEY (protein_id) REFERENCES Protein(id),
    FOREIGN KEY (strain_id) REFERENCES StrainOrCellLine(id),
    FOREIGN KEY (condition_id) REFERENCES Condition(id),
    FOREIGN KEY (capture_setting_id) REFERENCES CaptureSetting(id),
    FOREIGN KEY (user_id) REFERENCES User(id)
);

CREATE TABLE IF NOT EXISTS File (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    field_of_view TEXT,
    file_type TEXT,
    file_path TEXT,
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);

""")

conn.commit()
conn.close()

print("Database and tables created successfully.")
