# Dataloader for Image Classifcation Task
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# Define normalization parameters
norm_mean = [0.4828895122298728, 0.4448394893850807, 0.39566558230789783]
norm_std = [0.25925664613996574, 0.2532760018681693, 0.25981017205097917]

# Change the dim in Resize for desired shape
oxford_iiit_pet_transform_train = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize images to 384*384
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(norm_mean, norm_std),  # Normalisation
])

# Define the Oxford-IIIT Pet Dataset Class for Classification
class OxfordIIITPetDataset(Dataset):
    def __init__(self, img_dir, list_file, transform=None, mode=None, is_defense = False):
        self.img_dir = img_dir
        self.transform = transform
        self.img_list = []
        self.mode = mode
        self.is_defense = is_defense
        # Parse the list file (list.txt)
        with open(list_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if(parts[0][0]=='#'):
                    continue
                img_name = parts[0]
                class_id = int(parts[1])  # 1:37 Class IDs (1-25: Cats, 26-37: Dogs)
                self.img_list.append((img_name, class_id))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name, class_id = self.img_list[idx]

        # Load the image path
        img_path = os.path.join(self.img_dir, img_name + '.jpg')  # Ensure it's a valid extension
            
        # Read the image data
        image = Image.open(img_path).convert('RGB')

        # Apply transformations (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)

        # Return the image tensor and class_id
        # sending class_id - 1 since the classes need to [0,num_classes-1] but here classes are [1,num_classes]
        return image, class_id - 1


if __name__ == "__main__":
    # Define the transformations for preprocessing (resize, normalize, etc.)
    transform = oxford_iiit_pet_transform_train

    # Considering that the images are extracted in the oxford-iiit-pet folder
    img_dir = './oxford-iiit-pet/images'  # Path to image directory
    trimap_dir = './oxford-iiit-pet/annotations/trimaps'  # Path to trimap annotations
    xml_dir = './oxford-iiit-pet/annotations/xmls'  # Path to bounding box XML files
    list_file = './oxford-iiit-pet/annotations/test.txt'  # Path to list.txt

    # Create the dataset instance for classification
    dataset = OxfordIIITPetDataset(img_dir, list_file, transform=transform)

    # Create DataLoader for batching and shuffling
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Iterate over DataLoader to test it
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Labels range: {labels.min()} - {labels.max()}")
        print(f"Batch {batch_idx + 1}")
        print(f"Images shape: {images.shape}")
        print(f"Labels: {labels}")
        break  # Only check one batch for testing