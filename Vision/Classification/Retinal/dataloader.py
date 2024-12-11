# Dataloader for Imagenet-1K for iamge classification task
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys

sys.path.append(os.path.abspath('../'))
from split import split_dataset

base_dir = "."
image_size = 224

class_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.9),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15, expand=False, center=None),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))  # Simulate zoom
])

transform_train = transforms.Compose(
    [transforms.Resize([image_size, image_size]),
    transforms.CenterCrop(image_size),
    transforms.RandomApply([class_transforms], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose(
    [transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if __name__=="__main__":
    # Define the root directory for ImageNet dataset
    root_dir = os.path.join(base_dir, 'dataset')
    batch_size = 4
    num_workers = 4

    train_dir = os.path.join(root_dir)
    test_dir = os.path.join(root_dir)

    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = ImageFolder(root=test_dir, transform=transform_test)

    # Create DataLoader for each dataset
    train_dataset, test_dataset = split_dataset(train_dataset, train_percentage=0.70, batch_size=batch_size, num_workers=num_workers)
    test_dataset, valid_dataset = split_dataset(test_dataset, train_percentage=0.50, batch_size=batch_size, num_workers=num_workers)
    
    # Create DataLoaders for both sets
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
        print(valid_dataset.__len__())
        break

    for images, labels in test_loader:
        print(f'Test Batch - Images: {images.shape}, Labels (should be -1 or empty): {labels}')
        print(test_dataset.__len__())
        break