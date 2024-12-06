import tarfile

# EXTRACTING THE ANNOTATIONS
tar_file_path = 'annotations.tar.gz'  # Replace with your tar file path


extract_to_path = './oxford-iiit-pet/'  # Replace with your desired extraction directory

with tarfile.open(tar_file_path, 'r') as tar:
    tar.extractall(path=extract_to_path)
    print(f"Extracted all files to {extract_to_path}")
    

# EXTRACTING THE IMAGES   
tar_file_path = '/images.tar.gz'  # Replace with your tar file path

with tarfile.open(tar_file_path, 'r') as tar:
    tar.extractall(path=extract_to_path)
    print(f"Extracted all files to {extract_to_path}")