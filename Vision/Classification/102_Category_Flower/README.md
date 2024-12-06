# Dataset: 102 Category Flower

Welcome to the **102 Category Flower** dataset folder! This directory contains all the necessary resources to download, prepare, and load the dataset for your machine learning tasks.

---

## 🚀 Getting Started

To get started with **102 Category Flower** dataset, follow these steps:


### 1. Requirements

You might need to download some python packages for running the `dataprep.py` and `dataloader.py`. Run the following commands to download required python packages.

```bash
pip install tarfile torch torchvision torchaudio pillow
```

### 2. Download the Dataset

You can download the dataset from the official source:

- [Download **102 Category Flower** Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)

#### Instructions:
1. Visit the **Downloads** section on the webpage.
2. You will find three direct links to download the following files:
   - **Dataset Images**: This will download a file named `102flowers.tgz` containing the dataset images.
   - **Image Labels**: This will download a `.mat` file titled `imagelabels.mat`, which contains the labels for the images.
   - **Data Splits**: This will download a `.mat` file titled `setid.mat`, which contains the dataset splits.
3. Optionally, download the `README` file available on the website for more details about the dataset structure and usage.

#### Note:
You may need to agree to the dataset's terms of use before downloading the files.


### 3. Prepare the Dataset

Once the dataset is downloaded, run the file **`dataprep.py`** to prepare the data for use. This script handles tasks like unzipping archives, organizing files, and converting the dataset into the proper format for machine learning.

Run the following command to begin preparing the dataset:
```bash
python dataprep.py
```
Here dataprep mainly involves decompressing the tar files for the images present in the dataset.

### 4. Dataloader

Once the dataset is prepared, the dataloader for using the dataset is present in **`dataloader.py`**. This handles loading, transforming, and batching the dataset.

For a sample output of input shapes, run the following
```bash
python dataloader.py
```

You will get a sample image shapes, labels when you run this script

---