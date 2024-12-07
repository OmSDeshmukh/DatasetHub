# Cifar-10 dataloader for classification task
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import torchvision.transforms as transforms


Cifar_10_transform_train = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize images to 384*384
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

Cifar_10_transform_test = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize images to 384*384
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Custom Dataset for loading CIFAR-10 images and labels
class CIFAR10Dataset(Dataset):
    def __init__(self, image_dir, mode, transform=None, is_defense = False):
        '''
        mode: Whether test, train, val
        '''
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.mode = mode
        self.is_defense = is_defense
        # Go through each label folder and gather image paths and labels
        for label in os.listdir(image_dir):
            label_dir = os.path.join(image_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(int(label))  # Folder name is the label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        # Apply any transformations (if provided)
        if self.transform:
            image = self.transform(image)
        
        return image, label


if __name__ == "__main__":
    # For train(For val, you can split the train itself into 90-10 split)
    # Create dataset and dataloader
    image_dir = './Cifar-10/train'
    dataset = CIFAR10Dataset(image_dir, transform=Cifar_10_transform_train, mode="train")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")  # Should be (batch_size, 3, 384, 384)
        print(f"Labels: {labels}")  # Tensor of labels
        break  # Only check one batch for testing
    
    
    # For testing
    image_dir = './Cifar-10/test'
    dataset = CIFAR10Dataset(image_dir, transform=Cifar_10_transform_test, mode="test")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")  # Should be (batch_size, 3, 384, 384)
        print(f"Labels: {labels}")  # Tensor of labels
        break  # Only check one batch for testing
    