from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch

def split_dataset(dataset, train_percentage=0.8, batch_size=4, num_workers=1, seed=42):
    """
    Splits a PyTorch dataset into training and validation sets based on a specified percentage.

    Args:
        dataset (Dataset): The dataset to be split.
        train_percentage (float): The percentage of the dataset to use for training (between 0 and 1).

    Returns:
        DataLoader: DataLoader for the training set.
        DataLoader: DataLoader for the validation set.
    """
    # Ensure the percentage is valid
    if not (0 < train_percentage < 1):
        raise ValueError("train_percentage must be between 0 and 1")

    # Calculate the sizes of the datasets
    total_size = len(dataset)
    train_size = int(total_size * train_percentage)
    val_size = total_size - train_size

    # Set the seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Split the dataset
    train_dataset, valid_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    return train_dataset, valid_dataset