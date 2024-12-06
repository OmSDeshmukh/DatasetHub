import tarfile

# EXTRACT THE DATASET IMAGES
tar_file_path = '/102flowers.tgz'  # Replace with your tar file path
extract_to_path = './102flowers/'  # Replace with your desired extraction directory

with tarfile.open(tar_file_path, 'r') as tar:
    tar.extractall(path=extract_to_path)
    print(f"Extracted all files to {extract_to_path}")