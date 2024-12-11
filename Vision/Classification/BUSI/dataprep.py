from zipfile import ZipFile
import os
import shutil

# base_dir = os.getenv('DLP_BASE_DIR')
base_dir = "."

# Path to the rar file
zip_file_path = os.path.join(base_dir, 'BreastUS.zip')

# Directory where the rar will be extracted
extract_to = base_dir

# Ensure the directory exists
os.makedirs(extract_to, exist_ok=True)

# Open and extract the rar file
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(path=extract_to)

print(f'Files extracted to {extract_to}')

# Segregate images and masks
# root_path = os.path.join(base_dir,'datasets','Dataset_BUSI_with_GT')
root_path = os.path.join(base_dir,'Dataset_BUSI_with_GT')
images_path = root_path + '_images'
masks_path = root_path + '_masks'

if not os.path.exists(images_path):
    os.mkdir(images_path)
if not os.path.exists(masks_path):
    os.mkdir(masks_path)

classes = os.listdir(root_path)

for data_class in classes:
    root_class_path = os.path.join(root_path, data_class)
    images_class_path = os.path.join(images_path, data_class)
    masks_class_path = os.path.join(masks_path, data_class)

    if not os.path.exists(images_class_path):
        os.mkdir(images_class_path)
    
    if not os.path.exists(masks_class_path):
        os.mkdir(masks_class_path)
    
    files = os.listdir(root_class_path)

    for file in files:
        if 'mask' in file:
            shutil.copyfile(os.path.join(root_class_path, file), os.path.join(masks_class_path, file))
        else:
            shutil.copyfile(os.path.join(root_class_path, file), os.path.join(images_class_path, file))