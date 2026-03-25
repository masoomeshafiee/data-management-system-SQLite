import os
import csv
import shutil

def rename_files_from_metadata(metadata_csv, base_folder, dry_run=True, copy_to_subfolder=True):
    with open(metadata_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            folder = os.path.join(base_folder, row['relative_folder'])
            old_path = os.path.join(folder, row['file_name'])
            if not os.path.isfile(old_path):
                print(f"[SKIP] File not found: {old_path}")
                continue
            
            # Build new filename
            ext = os.path.splitext(row['file_name'])[1]  # Keep original extension
            new_name = f"{row['experiment_id']}_{row['organism']}_{row['protein']}_{row['condition']}_{row['capture_type']}_{row['field_of_view']}_{row['file_type']}{ext}"
            new_name = new_name.replace(' ', '_').replace('%', 'pct')
            
            if copy_to_subfolder:
                target_folder = os.path.join(folder, 'renamed_files')
                os.makedirs(target_folder, exist_ok=True)
                new_path = os.path.join(target_folder, new_name)
            else:
                new_path = os.path.join(folder, new_name)
            
            # Show action
            print(f"\nPreparing to {'COPY' if copy_to_subfolder else 'RENAME'}:")
            print(f"FROM: {old_path}\nTO:   {new_path}")
            
            # Dry-run mode
            if dry_run:
                print("→ Dry run enabled — no changes made.")
            else:
                if copy_to_subfolder:
                    shutil.copy2(old_path, new_path)
                    print("→ File copied.")
                else:
                    shutil.move(old_path, new_path)
                    print("→ File renamed.")
    
    print("\n=== Process completed. ===")

# Example usage
if __name__ == "__main__":
    metadata_csv = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/confocal/metadata_template.csv"  # <-- Your filled metadata file
    base_folder = "/Volumes/Masoumeh/Masoumeh/Masoumeh_data/1-Rfa1/confocal"  # <-- Root folder where the files are located
    
    rename_files_from_metadata(
        metadata_csv,
        base_folder,
        dry_run=False,             # <-- Set False to actually rename/copy
        copy_to_subfolder=False    # <-- True: copy into 'renamed_files'; False: rename in place
    )
