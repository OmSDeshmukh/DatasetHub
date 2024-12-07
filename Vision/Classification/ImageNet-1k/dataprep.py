from datasets import load_dataset
# huggingface-cli login-<your huggingface API key>
# If the dataset is gated/private, make sure you have run huggingface-cli login
dataset = load_dataset("ILSVRC/imagenet-1k",cache_dir='./ImageNet-1k')
print(dataset)

import os
import torch
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

def save_image(args):
    i, split, tensor_dataset, root_dir = args
    label = tensor_dataset[split][i]['label']
    
    class_dir = os.path.join(root_dir, split, str(label))
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f"{i}.JPEG")
    
    # Skip if the image has already been saved
    if os.path.exists(image_path):
        return
    
    image = tensor_dataset[split][i]['image']
    
    # Convert image to RGB if it is in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image.save(image_path, "JPEG")  # Save PIL image as JPEG

class PILImageFolder(torch.utils.data.Dataset):
    def __init__(self, tensor_dataset, root_dir):
        self.tensor_dataset = tensor_dataset
        self.root_dir = root_dir

        # Create directory structure
        for split in tensor_dataset.keys():

                split_dir = os.path.join(root_dir, split)
                os.makedirs(split_dir, exist_ok=True)

                # Create arguments for the save_image function
                args_list = [(i, split, tensor_dataset, root_dir) for i in range(len(tensor_dataset[split]))]

                # Parallelize saving images using multiprocessing Pool
                with Pool() as pool:
                    list(tqdm(pool.imap(save_image, args_list), total=len(tensor_dataset[split]), desc=f"Saving {split} images"))


root_dir = "./ImageNet-1k"
image_folder_dataset = PILImageFolder(dataset, root_dir)