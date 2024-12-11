import zipfile
import os

base_dir = "."

# Path to the zip file
zip_file_path = os.path.join(base_dir, 'archive.zip')

# Directory where the zip will be extracted
extract_to = base_dir

# Ensure the directory exists
os.makedirs(extract_to, exist_ok=True)

# Open and extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f'Files extracted to {extract_to}')