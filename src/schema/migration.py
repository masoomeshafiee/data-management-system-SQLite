import sqlite3
DB_PATH = "/Users/masoomeshafiee/Projects/data_organization/data-management-system-SQLite/db/Reyes_lab_data.db"
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON;")
c = conn.cursor()
#c.executescript("""ALTER TABLE Experiment
#    ADD COLUMN is_valid TEXT;""")

#c.execute("""ALTER TABLE Experiment
#   ADD COLUMN replicate INTEGER;""")

#c.execute("""ALTER TABLE Experiment
#    ADD COLUMN comment TEXT;""")

#c.execute("""ALTER TABLE User RENAME TO User_old;""")
#c.execute("""DELETE FROM Organism WHERE id = 2;""")
#c.execute("""DROP TABLE IF EXISTS User;""")
'''c.executescript("""
CREATE TABLE User (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT UNIQUE,
    UNIQUE(name, last_name)
    );
INSERT INTO User (id, name, last_name, email)
SELECT id, name, last_name, email FROM User_old;
                
DROP TABLE User_old;""")
'''

                
'''c.execute("""ALTER TABLE Condition_old RENAME TO Condition_old2;""")

c.executescript("""
CREATE TABLE Condition (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    concentration_value REAL,
    concentration_unit TEXT
    );
INSERT INTO Condition (id, name, concentration_value, concentration_unit)
SELECT id, name, concentration_value, concentration_unit FROM Condition_old2;
                
DROP TABLE Condition_old2;
                """)

             
c.executescript("""DELETE FROM Condition;
DELETE FROM EXperiment;
DELETE FROM CaptureSetting;
DELETE FROM StrainOrCellLine;
DELETE FROM Protein;
DELETE FROM Organism;
DELETE FROM User;
DELETE FROM RawFile;
DELETE FROM TrackingFiles;
DELETE FROM Masks;
DELETE FROM sqlite_sequence WHERE name='User';
DELETE FROM sqlite_sequence WHERE name='Experiment';
DELETE FROM sqlite_sequence WHERE name='Condition';
DELETE FROM sqlite_sequence WHERE name='CaptureSetting';
DELETE FROM sqlite_sequence WHERE name='StrainOrCellLine';
DELETE FROM sqlite_sequence WHERE name='Protein';
DELETE FROM sqlite_sequence WHERE name='Organism';
DELETE FROM sqlite_sequence WHERE name='RawFile';
DELETE FROM sqlite_sequence WHERE name='TrackingFiles';
DELETE FROM sqlite_sequence WHERE name='Masks';
""")

'''

#c.execute("""ALTER TABLE File RENAME TO RawFile;""")
'''
c.executescript("""
CREATE TABLE IF NOT EXISTS TrackingFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    file_path TEXT,
    file_type TEXT,
    field_of_view TEXT,
    theshold INTEGER,
    linking_distance FLOAT,
    Gap_closing_distance FLOAT,
    max_frame_gap INTEGER,
    Experiment_id INTEGER NOT NULL,
    FOREIGN KEY (Experiment_id) REFERENCES Experiment(id)
                );
    
CREATE TABLE IF NOT EXISTS Masks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mask_name TEXT NOT NULL,
    mask_path TEXT,
    field_of_view TEXT,
    mask_type TEXT, -- e.g., 'nuclei', 'Brightfield'
    file_type TEXT, -- e.g., 'mask.png, 'seg.npy', outline.txt
    Segmentation_method TEXT, -- e.g., 'Cellpose', 'Thresholding'
    Segmentation_parameters TEXT, -- e.g., 'cellpose_model=cyto', 'threshold=0.5' decide later ( maybe a JSON string)
    Experiment_id INTEGER NOT NULL,
    FOREIGN KEY (Experiment_id) REFERENCES Experiment(id)
                );

CREATE TABLE IF NOT EXISTS AnalysisFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    file_path TEXT,
    file_type TEXT, -- e.g., 'NOQ.mat', 'Q.mat', etc.
    field_of_view TEXT);
    
                #*******************************
CREATE TABLE IF NOT EXISTS AnalysisResults (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_type TEXT, -- e.g., 'intensity_over_time', 'dwell time', 'bound_fraction', copy number'
    result_path TEXT,
    result_value FLOAT,
    sample_size INTEGER,
    standard_error FLOAT,
    Analysis_method TEXT, -- e.g., 'Bound2Learn', 'MSD', 'SMOL'
    analysis_parameters TEXT, -- e.g., 'threshold=0.5, min_track_length=10' decide later ( maybe a JSON string)
    Analysis_file_path TEXT -- Path to the analysis file
);
                

CREATE TABLE IF NOT EXISTS ExperimentAnalysisFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    analysis_file_id INTEGER NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id),
    FOREIGN KEY (analysis_file_id) REFERENCES AnalysisFiles(id),
    UNIQUE (experiment_id, analysis_file_id)
);

CREATE TABLE IF NOT EXISTS AnalysisResultExperiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_result_id INTEGER NOT NULL,
    experiment_id INTEGER NOT NULL,
    FOREIGN KEY (analysis_result_id) REFERENCES AnalysisResults(id),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id),
    UNIQUE (analysis_result_id, experiment_id)
);
                """)


'''
'''
c.executescript("""DELETE FROM Condition;
DELETE FROM EXperiment;
DELETE FROM CaptureSetting;
DELETE FROM StrainOrCellLine;
DELETE FROM Protein;
DELETE FROM Organism;
DELETE FROM User;
DELETE FROM TrackingFiles;
DELETE FROM Masks;
DELETE FROM AnalysisFiles;
DELETE FROM AnalysisResults;
DELETE FROM ExperimentAnalysisFiles;
DELETE FROM AnalysisResultExperiments;
DELETE FROM RawFiles;
DELETE FROM sqlite_sequence WHERE name='RawFiles';
DELETE FROM sqlite_sequence WHERE name='User';
DELETE FROM sqlite_sequence WHERE name='Experiment';
DELETE FROM sqlite_sequence WHERE name='Condition';
DELETE FROM sqlite_sequence WHERE name='CaptureSetting';
DELETE FROM sqlite_sequence WHERE name='StrainOrCellLine';
DELETE FROM sqlite_sequence WHERE name='Protein';
DELETE FROM sqlite_sequence WHERE name='Organism';
DELETE FROM sqlite_sequence WHERE name='TrackingFiles';
DELETE FROM sqlite_sequence WHERE name='Masks';
DELETE FROM sqlite_sequence WHERE name='AnalysisFiles';
DELETE FROM sqlite_sequence WHERE name='AnalysisResults';
DELETE FROM sqlite_sequence WHERE name='ExperimentAnalysisFiles';
DELETE FROM sqlite_sequence WHERE name='AnalysisResultExperiments';
""")
'''
'''

c.execute("""ALTER TABLE  Experiment_old67
    RENAME TO Experiment_old7;""")
c.executescript("""
CREATE TABLE IF NOT EXISTS Experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organism_id INTEGER,
    protein_id INTEGER,
    strain_id INTEGER,
    condition_id INTEGER,
    capture_setting_id INTEGER,
    user_id INTEGER,
    date TEXT,
    replicate INTEGER,  -- New column for replicate number
    is_valid TEXT,  -- New column for validity
    comment TEXT,  -- New column for comments
    experiment_path TEXT,
    FOREIGN KEY (organism_id) REFERENCES Organism(id),
    FOREIGN KEY (protein_id) REFERENCES Protein(id),
    FOREIGN KEY (strain_id) REFERENCES StrainOrCellLine(id),
    FOREIGN KEY (condition_id) REFERENCES Condition(id),
    FOREIGN KEY (capture_setting_id) REFERENCES CaptureSetting(id),
    FOREIGN KEY (user_id) REFERENCES User(id),
    UNIQUE (organism_id, protein_id, strain_id, condition_id, capture_setting_id, user_id, date, replicate));
INSERT INTO Experiment (id, organism_id, protein_id, strain_id, condition_id, capture_setting_id, user_id, date, replicate, is_valid, comment, experiment_path)
SELECT id, organism_id, protein_id, strain_id, condition_id, capture_setting_id, user_id, date, replicate, is_valid, comment, experiment_path FROM Experiment_old7;          
DROP TABLE Experiment_old7;""")
'''
# -------------------
'''
c.execute("""DROP TABLE IF EXISTS RawFile;""")
c.executescript("""
CREATE TABLE IF NOT EXISTS RawFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Experiment_id INTEGER NOT NULL,
    file_name TEXT NOT NULL,
    field_of_view TEXT,
    file_type TEXT,
    file_path TEXT,
    FOREIGN KEY (Experiment_id) REFERENCES Experiment(id)
                );
""")
'''
#---------- ADD unique constraints ----------------
# --- RawFiles ---
'''
c.execute("""
    SELECT *
    FROM RawFiles_old2
    WHERE (experiment_id, file_name, field_of_view) IN (
        SELECT experiment_id, file_name, field_of_view
        FROM RawFiles_old2
        GROUP BY experiment_id, file_name, field_of_view
        HAVING COUNT(*) > 1
    )
    ORDER BY experiment_id, file_name, field_of_view;
""")
rows = c.fetchall()
print(len(rows))
for row in rows:
    print(row)
'''

'''
c.executescript(""" CREATE TABLE IF NOT EXISTS RawFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    file_name TEXT COLLATE NOCASE NOT NULL,
    field_of_view TEXT COLLATE NOCASE,
    file_type TEXT COLLATE NOCASE,
    file_path TEXT COLLATE NOCASE,
    UNIQUE (experiment_id, file_name, field_of_view, file_type),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);
""")

# --- TrackingFiles ---
c.executescript("""
ALTER TABLE TrackingFiles RENAME TO TrackingFiles_old;

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
    UNIQUE (experiment_id, file_name, field_of_view, file_type, threshold, linking_distance, gap_closing_distance, max_frame_gap),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);

INSERT INTO TrackingFiles (id, experiment_id, file_name, field_of_view, file_type, file_path,
                           threshold, linking_distance, gap_closing_distance, max_frame_gap)
SELECT id, experiment_id, file_name, field_of_view, file_type, file_path,
       threshold, linking_distance, gap_closing_distance, max_frame_gap
FROM TrackingFiles_old;

DROP TABLE TrackingFiles_old;
""")
#--- Masks ---
c.executescript("""
DROP TABLE Masks;
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
    UNIQUE (experiment_id, mask_name, field_of_view, mask_type, file_type, segmentation_method, segmentation_parameters),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id)
);
""")

# --- User ---
c.executescript("""
ALTER TABLE User RENAME TO User_old;

CREATE TABLE User (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT COLLATE NOCASE NOT NULL,
    last_name TEXT COLLATE NOCASE,
    email TEXT COLLATE NOCASE UNIQUE,
    UNIQUE(name, last_name)
);

INSERT INTO User (id, name, last_name, email)
SELECT id, name, last_name, email FROM User_old;

DROP TABLE User_old;
""")

# --- Condition ---
c.executescript("""
ALTER TABLE Condition RENAME TO Condition_old;

CREATE TABLE Condition (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT COLLATE NOCASE NOT NULL,
    concentration_value FLOAT,
    concentration_unit TEXT COLLATE NOCASE,
    UNIQUE (name, concentration_value, concentration_unit)
);

INSERT INTO Condition (id, name, concentration_value, concentration_unit)
SELECT id, name, concentration_value, concentration_unit FROM Condition_old;

DROP TABLE Condition_old;
""")
# ---------AnalysisFiles -----------
c.executescript("""
ALTER TABLE AnalysisFiles RENAME TO AnalysisFiles_old;
CREATE TABLE AnalysisFiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT COLLATE NOCASE NOT NULL,
    file_path TEXT COLLATE NOCASE,
    file_type TEXT COLLATE NOCASE,
    field_of_view TEXT COLLATE NOCASE,
    UNIQUE (file_name, file_path, file_type, field_of_view)
);
INSERT INTO AnalysisFiles (id, file_name, file_path, file_type, field_of_view)
SELECT id, file_name, file_path, file_type, field_of_view
FROM AnalysisFiles_old;
DROP TABLE AnalysisFiles_old;
""")
'''
'''
c.executescript("""DROP TABLE IF EXISTS TrackingFiles;
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
    UNIQUE (experiment_id, file_name, field_of_view, file_type, threshold, linking_distance, gap_closing_distance, max_frame_gap),
    FOREIGN KEY (experiment_id) REFERENCES Experiment(id));""")
'''
# c.executescript(""" DELETE FROM Experiment WHERE id=53; DELETE FROM Experiment WHERE id=54; DELETE FROM Experiment WHERE id=55; DELETE FROM Experiment WHERE id=56; DELETE FROM CaptureSetting WHERE id=6; DELETE FROM CaptureSetting WHERE id=7; DELETE FROM Condition WHERE id=6; DELETE FROM Condition WHERE id=7;""")

#------------------------------------------------------------
#-- Dimension / FK target tables: identities used in lookups
#------------------------------------------------------------

#-- Organism: identified by name
'''
c.executescript("""CREATE UNIQUE INDEX IF NOT EXISTS uq_Organism_identity
ON Organism(name); -- Protein: identified by name
CREATE UNIQUE INDEX IF NOT EXISTS uq_Protein_identity ON Protein(name);
-- Strain or cell line: identified by name
CREATE UNIQUE INDEX IF NOT EXISTS uq_StrainOrCellLine_identity
ON StrainOrCellLine(name);
-- User: identified by email (safer than name)
CREATE UNIQUE INDEX IF NOT EXISTS uq_User_identity
ON User(email);
-- Condition: typical identity (name, concentration_value, concentration_unit)
CREATE UNIQUE INDEX IF NOT EXISTS uq_Condition_identity
ON Condition(name, concentration_value, concentration_unit);
-- CaptureSetting: identity used in your updates
CREATE UNIQUE INDEX IF NOT EXISTS uq_CaptureSetting_identity
ON CaptureSetting(capture_type, exposure_time, time_interval, dye_concentration_value);
""")
'''
#------------------------------------------------------------
#-- Main row identity (so you can match Experiments by naturals)
#------------------------------------------------------------

#-- If you want Experiments to be uniquely identified by naturals:
#-- date + replicate + all dependent FKs
'''
c.executescript("""CREATE UNIQUE INDEX IF NOT EXISTS uq_Experiment_identity
ON Experiment(
  date,
  replicate,
  user_id,
  capture_setting_id,
  condition_id,
  protein_id,
  organism_id,
  strain_id
);""")
'''
#------------------------------------------------------------
#-- Rename 'name' columns to more specific names
#------------------------------------------------------------
c.executescript("""ALTER TABLE Condition RENAME COLUMN name TO condition_name;
                ALTER TABLE Organism RENAME COLUMN name TO organism_name;
                ALTER TABLE Protein RENAME COLUMN name TO protein_name;
                ALTER TABLE StrainOrCellLine RENAME COLUMN name TO strain_name;
                ALTER TABLE User RENAME COLUMN name TO user_name;""")
conn.commit()
conn.close()
print("Database schema updated successfully.")