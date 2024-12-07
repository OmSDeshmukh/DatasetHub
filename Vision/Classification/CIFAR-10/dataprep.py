import os
import pickle
import numpy as np
from PIL import Image

# Function to unpickle the batch files
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Function to save the images from CIFAR-10 batches
def save_cifar10_images(batch_files, save_dir, test_save_dir):
    # Create directories if they don't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(test_save_dir):
        os.makedirs(test_save_dir)
    
    for batch_file in batch_files:
        # Load the batch
        batch_file_path = os.path.join("./Cifar-10", "cifar-10-batches-py", batch_file)
        batch = unpickle(batch_file_path)
        images = batch[b'data']  # 10000x3072 numpy array of uint8s
        labels = batch[b'labels']  # List of labels
        
        for i in range(images.shape[0]):
            # Extract the image
            img = images[i]
            img_r = img[0:1024].reshape(32, 32)  # Red channel
            img_g = img[1024:2048].reshape(32, 32)  # Green channel
            img_b = img[2048:].reshape(32, 32)  # Blue channel
            img_rgb = np.dstack((img_r, img_g, img_b))  # Combine R, G, B
            
            # Create a folder for each label if it doesn't exist
            label = labels[i]
            label_dir = os.path.join(save_dir, str(label))  # For training images
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            
            # Save the image as a PNG file in the training directory
            img_save_path = os.path.join(label_dir, f'{batch_file.split("/")[-1]}_img_{i}.png')
            img_rgb = Image.fromarray(img_rgb)
            img_rgb.save(img_save_path)

        print(f"Processed batch: {batch_file}")

# Specify the CIFAR-10 batch files and where to save the images
train_batch_files = [
    'data_batch_1', 'data_batch_2', 'data_batch_3',
    'data_batch_4', 'data_batch_5'
]
test_batch_file = 'test_batch'  # Only one test batch

# Save CIFAR-10 training images to this directory
train_save_dir = './Cifar-10/train'

# Save CIFAR-10 test images to this directory
test_save_dir = './Cifar-10/test'

# Generate and save CIFAR-10 training images and labels
save_cifar10_images(train_batch_files, train_save_dir, test_save_dir)

# Save CIFAR-10 test images
save_cifar10_images([test_batch_file], test_save_dir, test_save_dir)