import rarfile
import os

base_dir = "."

# Path to the rar file
rar_file_path = os.path.join(base_dir, 'fold1_seperated_2_classes.rar')

# Directory where the rar will be extracted
extract_to = base_dir

# Ensure the directory exists
os.makedirs(extract_to, exist_ok=True)

# Open and extract the rar file
with rarfile.RarFile(rar_file_path) as rar_ref:
    rar_ref.extractall(extract_to)

print(f'Files extracted to {extract_to}')