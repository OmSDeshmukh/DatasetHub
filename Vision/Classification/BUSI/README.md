# Dataset: BUSI

Welcome to the **BUSI** dataset folder! This directory contains all the necessary resources to download, prepare, and load the dataset for your machine learning tasks.

---

## Background

The BUSI dataset contains images of breast cancer using ultrasound scan. 
It contains 780 ultrasound microscopic images: 437 benign, 210 malignant, and 133 normal cases.

## 🚀 Getting Started

To get started with **BUSI** dataset, follow these steps:


### 1. Requirements

You might need to download some python packages for running the `dataprep.py` and `dataloader.py`. Run the following commands to download required python packages.

```bash
pip install torch torchvision torchaudio pillow numpy tqdm multiprocess shutil zipfile
```

### 2. Download the Dataset

You can download the dataset from the official source by creating an official login on the website and agreeing to its terms and conditions of use.
This might require purchasing a subscription.
- [Download BUSI Dataset](https://ieee-dataport.org/documents/four-public-datasets-explainable-medical-image-classifications)

- [Here](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) I provide another way to download the dataset from Kaggle. 


#### Instructions:
When using the Kaggle method, the code for data preparation is present in the `dataprep.py` file itself.

#### Note:
You may need to agree to the dataset's terms of use before downloading the files.


### 3. Prepare the Dataset

Run the file **`dataprep.py`** to prepare the data for use. This script handles tasks like unzipping, organizing files and converting the dataset into the proper format for machine learning.

Run the following command to begin preparing the dataset:
```bash
python dataprep.py
```

Here dataprep mainly involves unzipping the files and segregating them into images and masks.


### 4. Dataloader

Once the dataset is prepared, the dataloader for using the dataset is present in **`dataloader.py`**. This handles loading, transforming, and batching the dataset.

For a sample output of input shapes, run the following
```bash
python dataloader.py
```

You will get a sample image shapes, labels when you run this script.

---