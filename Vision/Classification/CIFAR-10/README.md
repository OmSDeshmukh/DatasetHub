# Dataset: CIFAR-10

Welcome to the **CIFAR-10** dataset folder! This directory contains all the necessary resources to download, prepare, and load the dataset for your machine learning tasks.

---

## 🚀 Getting Started

To get started with **CIFAR-10** dataset, follow these steps:


### 1. Requirements

You might need to download some python packages for running the `dataprep.py` and `dataloader.py`. Run the following commands to download required python packages.

```bash
pip install torch torchvision torchaudio pillow numpy
```

### 2. Download the Dataset

You can download the dataset from the official source:

- [Download **CIFAR-10** Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

#### Instructions:
Visit the **Downloads** section on the webpage under **The CIFAR-10 dataset** section. Choose the `CIFAR-10 python version` link to download the required files. This will download `cifar-10-python.tar.gz` file which will be prepared using the `dataprep.py` file

#### Note:
You may need to agree to the dataset's terms of use before downloading the files.


### 3. Prepare the Dataset

Once the dataset is downloaded, run the file **`dataprep.py`** to prepare the data for use. This script handles tasks like unzipping archives, organizing files, and converting the dataset into the proper format for machine learning.

Run the following command to begin preparing the dataset:
```bash
python dataprep.py
```
Here dataprep mainly involves decompressing the tar files for the images present in the dataset. This will create two folders `train` and `test` to store the required images. Also the unzipped folder by the name `cifar-10-batches-py` does contain a readme file if needed for further investigation of the dataset.

### 4. Dataloader

Once the dataset is prepared, the dataloader for using the dataset is present in **`dataloader.py`**. This handles loading, transforming, and batching the dataset.

For a sample output of input shapes, run the following
```bash
python dataloader.py
```

You will get a sample image shapes, labels when you run this script

---