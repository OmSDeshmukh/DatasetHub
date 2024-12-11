from zipfile import ZipFile
import os

base_dir = "."

# Path to the rar file
zip_file_path = os.path.join(base_dir, 'Retinal.zip')

# Directory where the rar will be extracted
extract_to = base_dir

# Ensure the directory exists
os.makedirs(extract_to, exist_ok=True)

# Open and extract the rar file
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(path=extract_to)

print(f'Files extracted to {extract_to}')