import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import cv2

base_dir = "."
image_size = 224

# Define transformations for the images
data_transform = {
"train": transforms.Compose([transforms.RandomResizedCrop(image_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
"val": transforms.Compose([transforms.Resize((image_size, image_size)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

transform_train = data_transform['train']
transform_test = data_transform['val']


# {'NORMAL': 0, 'PNEUMONIA': 1}
if __name__=="__main__":
    # Define the root directory for ImageNet dataset
    root_dir = os.path.join(base_dir, 'chest_xray')
    batch_size = 4
    num_workers = 4

    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')

    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    valid_dataset = ImageFolder(root=val_dir, transform=transform_test)
    test_dataset = ImageFolder(root=test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
        print(f'Test Batch - Images: {images.shape}, Labels: {labels}')
        print(test_dataset.__len__())
        break