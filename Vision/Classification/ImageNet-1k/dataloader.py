# Dataloader for Imagenet-1K for iamge classification task
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet means/std
])


if __name__=="__main__":
    # Define the root directory for ImageNet dataset
    root_dir = './ImageNet-1k'  # Replace with actual path

    # Create datasets for train, val, and test
    # Train dataset (1000 classes)
    train_dir = os.path.join(root_dir, 'train')
    train_dataset = ImageFolder(root=train_dir, transform=transform)

    # Validation dataset (1000 classes)
    val_dir = os.path.join(root_dir, 'validation')
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    # Test dataset (single folder "-1", no labels)
    test_dir = os.path.join(root_dir, 'test')
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)


    # Test the loaders by loading a batch
    for images, labels in train_loader:
        print(f'Train Batch - Images: {images.shape}, Labels: {labels}')
        print(train_dataset.__len__())
        break

    for images, labels in val_loader:
        print(f'Validation Batch - Images: {images.shape}, Labels: {labels}')
        print(val_dataset.__len__())
        break

    for images, labels in test_loader:
        print(f'Test Batch - Images: {images.shape}, Labels: {labels}')
        print(test_dataset.__len__())
        break