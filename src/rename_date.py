import os
import re

# Folder containing the files
folder = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/bound fraction/HU/Rfa1_fast_549_60%_Sep21_HU_0.2"

# Regex: match DDMMYYYY at start, then the rest
pattern = re.compile(r"^(\d{2})(\d{2})(\d{4})(-.*)")

for filename in os.listdir(folder):
    match = pattern.match(filename)
    if match:
        day, month, year, rest = match.groups()
        new_name = f"{year}{month}{day}{rest}"
        
        old_path = os.path.join(folder, filename)
        new_path = os.path.join(folder, new_name)
        
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_name}")
