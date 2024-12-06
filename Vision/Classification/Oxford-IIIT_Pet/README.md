# Dataset: Oxford-IIIT Pet

Welcome to the **Oxford-IIIT Pet** dataset folder! This directory contains all the necessary resources to download, prepare, and load the dataset for your machine learning tasks.

---

## 🚀 Getting Started

To get started with **Oxford-IIIT Pet** dataset, follow these steps:


### 1. Requirements

You might need to download some python packages for running the `dataprep.py` and `dataloader.py`. Run the following to download required python packages.

```bash
pip install tarfile torch torchvision torchaudio pillow
```

### 2. Download the Dataset

You can download the dataset from the official source:

- [Download **Oxford-IIIT Pet**](<https://www.robots.ox.ac.uk/~vgg/data/pets/>)  dataset
- **Instructions**: You will find two direct links to download the images and annotations by the name `images.tar.gz` and `annotations.tar.gz` respectively. You may need to agree to the terms before downloading. Also, apart from the usual download link, there is also a torrent link availaible on the website for faster download.

### 3. Prepare the Dataset

Once the dataset is downloaded, run the file **`dataprep.py`** to prepare the data for use. This script handles tasks like unzipping archives, organizing files, and converting the dataset into the proper format for machine learning.

Run the following command to begin preparing the dataset:
```bash
python dataprep.py
```
Here dataprep mainly involves decompressing the tar files and putting them into repsective folders.

### 4. Dataloader

Once the dataset is prepared, the dataloader for using the dataset is present in **`dataloader.py`**. This handles loading, transforming, and batching the dataset.

For a sample output of input shapes, run the following
```bash
python dataloader.py
```

---