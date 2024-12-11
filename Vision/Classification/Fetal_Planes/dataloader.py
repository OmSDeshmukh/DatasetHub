import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys

sys.path.append(os.path.abspath('../'))
from split import split_dataset
from PIL import Image
import pandas as pd

base_dir = "."
image_size = 224

classes_to_id = {
    'Other': 0,
    'Maternal cervix': 1,
    'Fetal abdomen': 2,
    'Fetal brain': 3,
    'Fetal femur': 4,
    'Fetal thorax': 5
}

# Define a transform as described in the instructions
data_transform = {
    "train": transforms.Compose([
        transforms.Resize((image_size, image_size)),  # or (299, 299) for Inception model
        transforms.ToTensor(),  # Converts to tensor and scales image from [0, 255] to [0, 1]
        transforms.Normalize(mean=(0.5,), std=(0.5,))  # Normalize to [-1, 1] for grayscale
    ]),
    
    "val": transforms.Compose([
        transforms.Resize((image_size, image_size)),  # or (299, 299) for Inception model
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
}

transform_train = data_transform["train"]
transform_test = data_transform["val"]

# Submit this dataloader to Papers with code
class FETAL_PLANESDataset(Dataset):
    def __init__(self, root_dir, csv_file, image_size = 224, split='Train', transform=None, stack=None):
        '''
        Args:
            stack: Whether to stack images across dimensions
        
        '''
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.image_size = image_size
        self.image_paths = []
        self.transform = transform
        self.stack = stack
        
        df = pd.read_csv(self.csv_file, delimiter=';')
        value = 1 if split == 'Train' else 0
        
        for index, row in df.iterrows():
            image_dir_path = os.path.join(root_dir, "Images")
            if os.path.isdir(image_dir_path):
                if(row['Train ']==value):
                    class_idx = classes_to_id[row['Plane']]
                    
                    self.image_paths.append((os.path.join(image_dir_path, f'{row["Image_name"]}.png'), class_idx))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        
        # Read the image using OpenCV (BGR format)
        image = Image.open(img_path)
        if image.mode in ('RGB', 'RGBA', 'CMYK', 'LAB', 'YCbCr'):
            image = image.convert('L')
            
        # Check if the image is read properly
        if image is None:
            print(f"Error reading image: {img_path}")
            return torch.zeros((1, self.image_size, self.image_size)), -1  # Return a dummy tensor
        
        if(self.transform):
            image = self.transform(image)
            
        if(self.stack):
            image = image.repeat(3, 1, 1)
        
        return image, label  


# softmax cross-entropy loss and adam optimizer
# Early stopping patience 5
# Might have to extend this for further fine grained brain classification
if __name__=="__main__":
    # Define the root directory for ImageNet dataset
    root_dir = os.path.join(base_dir, 'FETAL_PLANES')
    csv_file = os.path.join(base_dir, 'FETAL_PLANES/FETAL_PLANES_DB_data.csv')
    batch_size = 4
    num_workers = 4

    train_dataset = FETAL_PLANESDataset(root_dir=root_dir, csv_file=csv_file, image_size=image_size, split='Train', transform=transform_train, stack=True)
    test_dataset = FETAL_PLANESDataset(root_dir=root_dir, csv_file=csv_file, image_size=image_size, split='Test', transform=transform_test, stack=True)

    # Create DataLoader for each dataset
    train_dataset, valid_dataset = split_dataset(train_dataset, train_percentage=0.90, batch_size=batch_size, num_workers=num_workers)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Test the loaders by loading a batch
    for images, labels in train_loader:
        print(f'Train Batch - Images: {images.shape}, Labels: {labels}')
        print(train_dataset.__len__())
        break
    
    for images, labels in valid_loader:
        print(f'Validation Batch - Images: {images.shape}, Labels: {labels}')
        break

    for images, labels in test_loader:
        print(f'Test Batch - Images: {images.shape}, Labels: {labels}')
        print(test_dataset.__len__())
        break