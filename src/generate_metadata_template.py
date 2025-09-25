import os
import csv
import re

def generate_metadata_template(folder_path, output_csv='metadata_template.csv'):
    rows = []
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith("._"):
                continue  # Skip hidden macOS metadata files
            if file.endswith(('.csv', '.tif', '.TIF', '.nd', 'npy', '.png')):
                #cell_id = file.split('-')[0] if '-' in file else 'unknown'
                # Auto-fill file_type based on filename
                file_type = extract_file_type(file)
                # Extract field of view from the file name
                field_of_view = extract_field_of_view(file)
                rows.append({
                    'relative_folder': os.path.relpath(root, folder_path),
                    'file_name': file,
                    'experiment_id': '',       # <-- date + replicate
                    'organism': 'yeast',            # <-- fill this in
                    'protein': 'Rad53',
                    'condition': 'untreated',           # <-- fill this in
                    'capture_type': 'confocal',        # e.g., 'fast' or 'long'
                    'field_of_view': field_of_view,      # e.g., '1', '2', etc.
                    'file_type': file_type,           # 'tracks', 'spots', 'mask', etc.
                })

    keys = rows[0].keys() if rows else []
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metadata template saved to: {output_csv}")

def extract_file_type(file):
    if 'spots' in file.lower():
        file_type = 'spots'
    elif 'tracks' in file.lower():
        file_type = 'tracks'
    elif 'masks' in file.lower():
        file_type = 'mask'
    elif 'npy' in file.lower():
        file_type = '_seg'
    elif 'max' in file.lower():
        file_type = 'projection'
    elif 'stream' in file.lower():
        file_type = 'video'
    elif file.lower().endswith('.nd'):
        file_type = 'nd'
    elif 'w2t2 gfp.tif' in file.lower():
        file_type = 'w2T2 GFP'
    elif 'w3t3 rfp cust.tif' in file.lower():
        file_type = 'w3T3 RFP CUST'
    elif 'w3t4 cy5'in file.lower():
        file_type = 'w3T4 Cy5'
    elif 'w1bf'in file.lower():
        file_type = 'w1BF'
    elif 'snap'in file.lower():
        file_type = 'BF_snapShot'
    elif 'gfp.tif' in file.lower():
        file_type = 'GFP'
    elif 'rfp.tif' in file.lower():
        file_type = 'RFP'
    elif 'subtracked' in file.lower():
        file_type = 'subtracked'
    elif 'w2dual-488-561-610-75'in file.lower():
        file_type = 'w2Dual-488-561-610-75'
    else:
        file_type = ''
    return file_type


def extract_field_of_view(file_name):
    # Case 1: Check for images
    image_match = re.search(r'([0-9]+)_w.*\.(TIF|tif|png|npy)$', file_name)
    if image_match:
        return image_match.group(1)
    
    # Case 2: Files starting with number and ending with '.nd'
    nd_match = re.search(r'([0-9]+)\.nd$', file_name)
    if nd_match:
        return nd_match.group(1)
    
    # Case 3: Check for 'StreamX'
    stream_match = re.search(r'Stream(\d+)', file_name, flags=re.IGNORECASE)
    if stream_match:
        return stream_match.group(1)
    
    # Case4: Check for 'SnapX'
    snap_match = re.search(r'Snap(\d+)', file_name, flags=re.IGNORECASE)
    if snap_match:
        return snap_match.group(1)
    
    return ''  # If no match found



# Example usage
if __name__ == "__main__":
    folder = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/confocal"  # <-- change this
    generate_metadata_template(folder, output_csv=os.path.join(folder, 'metadata_template.csv'))
