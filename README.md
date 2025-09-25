# Lab Data Management System (SQLite)

## Overview
This project is a **laboratory data management system** built around an **SQLite relational database**.  
It automates file organization, metadata extraction, database creation and migration, database management, data validation, and querying.  
The goal is to make laboratory research data **systematic, reproducible, FAIR-compliant, and easy to explore**.

---

## Features
- **Systematic File Naming** → script to rename raw files using a consistent naming convention.  
- **Metadata Extraction** → semi-automatic generation of metadata CSVs from filenames and folder structure.  
- **Database Management**  
  - SQLite database with normalized schema (Organism, Protein, Strain/CellLine, Condition, CaptureSetting, Experiment, etc.).
  - Scripts for initial creation, schema migrations, and validation.
- **Data Validation & Insertion**  
  - Validate metadata before inserting into the DB.  
  - Automatic  Bulk CSV-to-database insertion with logging. 
- **Querying Tools** 
  - collection of reusable query functions with a query-builder design (list experiments, count trends, detect anomalies, etc. ).  
- Command-line interface (Python Fire).  
- Natural language queries via LLM integration.

---

To be continued.. 