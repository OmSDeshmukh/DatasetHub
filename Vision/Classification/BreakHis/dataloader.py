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
quality = "40X"

normalise_test = {
    "40X": transforms.Normalize((0.8036, 0.6536, 0.7731), (0.1091, 0.1494, 0.1070)),
    "100X": transforms.Normalize((0.7997, 0.6358, 0.7745), (0.1270, 0.1779, 0.1187)),
    "200X": transforms.Normalize((0.7917, 0.6211, 0.7659), (0.1244, 0.1780, 0.1099)),
    "400X": transforms.Normalize((0.7651, 0.5763, 0.7524), (0.1397, 0.1966, 0.1112))
}

normalise_train = {
    "40X":  transforms.Normalize((0.8018, 0.6498, 0.7656), (0.1105, 0.1555, 0.1126)),
    "100X": transforms.Normalize((0.7940, 0.6362, 0.7699), (0.1279, 0.1791, 0.1167)),
    "200X": transforms.Normalize((0.7895, 0.6215, 0.7696), (0.1266, 0.1784, 0.1077)),
    "400X": transforms.Normalize((0.7562, 0.5913, 0.7426), (0.1425, 0.2017, 0.1156))
}

transform_train = transforms.Compose([
    transforms.Resize([image_size, image_size]),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    normalise_train[quality]
])

transform_test = transforms.Compose(
    [transforms.Resize([image_size, image_size]),
    transforms.ToTensor(),
    normalise_test[quality]
])


# {'Benign': 0, 'Malignant': 1}
if __name__=="__main__":
    # Define the root directory for ImageNet dataset
    root_dir = os.path.join(base_dir, 'fold1_seperated_2_classes')
    batch_size = 4
    num_workers = 4

    train_dir = os.path.join(root_dir, 'train', quality)
    test_dir = os.path.join(root_dir, 'test', quality)

    train_dataset = ImageFolder(root=train_dir, transform=transform_train)
    test_dataset = ImageFolder(root=test_dir, transform=transform_test)

    train_dataset, valid_dataset = split_dataset(train_dataset, train_percentage=0.95, batch_size=batch_size, num_workers=num_workers)

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