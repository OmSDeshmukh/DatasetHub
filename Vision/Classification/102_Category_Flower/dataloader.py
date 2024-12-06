# Dataloader for the 102flowers dataset for image classification task
import os
import scipy.io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


_102flowers_transform_train = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize images to 384*384
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.4353, 0.3773, 0.2872], std=[0.2966, 0.2455, 0.2698])  # Normalize using ImageNet means/std
])


class Flowers102Dataset(Dataset):
    def __init__(self, img_dir, labels_file, split_file, split='train', transform=None, is_defense = False):
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        self.is_defense = is_defense
        
        # Load image labels from imagelabels.mat
        labels_data = scipy.io.loadmat(labels_file)
        self.labels = labels_data['labels'][0]  # 1D array of labels

        # Load dataset splits from setid.mat
        split_data = scipy.io.loadmat(split_file)
        if self.split == 'test': # testa for testing using attacked dataset
            self.img_ids = split_data['tstid'][0]  # Test image ids
        elif self.split == 'valid':
            self.img_ids = split_data['valid'][0]  # Validation image ids
        elif self.split == 'train':
            self.img_ids = split_data['trnid'][0]  # Train image ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Get the image ID
        img_id = self.img_ids[idx]
        
        img_name = None
        
        # Load the image (image files are named "image_00001.jpg", "image_00002.jpg", etc.)
        img_name = f'image_{img_id:05d}.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        # Get the label (labels are 1-indexed in the .mat file, convert to 0-indexed for PyTorch)
        label = self.labels[img_id - 1] - 1

        # Apply any image transformations (e.g., resizing, normalization)
        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":

    # Define the transforms for image preprocessing
    transform = _102flowers_transform_train

    # Define the paths to the dataset
    img_dir = './102flowers/jpg'  # Path to extracted images
    labels_file = './102flowers/imagelabels.mat'  # Path to imagelabels.mat
    split_file = './102flowers/setid.mat'  # Path to setid.mat

    # Create dataset instances for training, validation, and test sets
    train_dataset = Flowers102Dataset(img_dir, labels_file, split_file, split='train', transform=transform)
    valid_dataset = Flowers102Dataset(img_dir, labels_file, split_file, split='valid', transform=transform)
    test_dataset = Flowers102Dataset(img_dir, labels_file, split_file, split='test', transform=transform)

    # Create DataLoader instances for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    print("Train Dataset", train_dataset.__len__())
    print("Valid Dataset", valid_dataset.__len__())
    print("Test Dataset", test_dataset.__len__())

    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")  # Tensor shape: (batch_size, 3, 224, 224)
        print(f"Labels: {labels}")  # Tensor of labels
        break  # Only check one batch for testing
    
    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(valid_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")  # Tensor shape: (batch_size, 3, 224, 224)
        print(f"Labels: {labels}")  # Tensor of labels
        break  # Only check one batch for testing
    
    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(test_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")  # Tensor shape: (batch_size, 3, 224, 224)
        print(f"Labels: {labels}")  # Tensor of labels
        break  # Only check one batch for testing