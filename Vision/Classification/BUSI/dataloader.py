# Cifar-10 dataloader for classification task
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys

sys.path.append(os.path.abspath('../'))
from split import split_dataset

all_class = ['malignant', 'normal','benign']
class_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.9),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15, expand=False, center=None),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))  # Simulate zoom
])

image_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomApply([class_transforms], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

BUSI_transform_train = data_transforms['train']
BUSI_transform_test = data_transforms['validation']


if __name__ == "__main__":
    root_dir = './Dataset_BUSI_with_GT_images'

    train_dir = os.path.join(root_dir)
    test_dir = os.path.join(root_dir)

    train_dataset = ImageFolder(root=train_dir, transform=BUSI_transform_train)
    test_dataset = ImageFolder(root=test_dir, transform=BUSI_transform_test)

    batch_size = 4
    num_workers = 4

    # Create DataLoader for each dataset
    train_dataset, test_dataset = split_dataset(train_dataset, train_percentage=0.80, batch_size=batch_size, num_workers=num_workers)
    train_dataset, valid_dataset = split_dataset(train_dataset, train_percentage=0.90, batch_size=batch_size, num_workers=num_workers)
    
    # Create DataLoaders for both sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")  # Should be (batch_size, 3, 224, 224)
        print(f"Labels: {labels}")  # Tensor of labels
        break  # Only check one batch for testing

    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(valid_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")  # Should be (batch_size, 3, 224, 224)
        print(f"Labels: {labels}")  # Tensor of labels
        break  # Only check one batch for testing
    
    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")  # Should be (batch_size, 3, 224, 224)
        print(f"Labels: {labels}")  # Tensor of labels
        break  # Only check one batch for testing
    