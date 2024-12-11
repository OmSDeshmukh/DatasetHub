# Dataset: CXRAY

Welcome to the **CXRAY** dataset folder! This directory contains all the necessary resources to download, prepare, and load the dataset for your machine learning tasks.

---

## Background

The Chest X-Ray Images (CXR) dataset contains 5228 images, which are divided into three classes: **covid-19** (1626), **pneumonia** (1800), and **normal** (1802).

## 🚀 Getting Started

To get started with **CXRAY** dataset, follow these steps:


### 1. Requirements

You might need to download some python packages for running the `dataprep.py` and `dataloader.py`. Run the following commands to download required python packages.

```bash
pip install torch torchvision torchaudio pillow numpy tqdm multiprocess zipfile
```

### 2. Download the Dataset

You can download the dataset from [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)


#### Instructions:
When using the above link, the code for data preparation is present in the `dataprep.py` file itself.

#### Note:
You may need to agree to the dataset's terms of use before downloading the files.


### 3. Prepare the Dataset

Run the file **`dataprep.py`** to prepare the data for use. This script unzips the downloaded file and extracts the contents to the desired directory.

Run the following command to begin preparing the dataset:
```bash
python dataprep.py
```


### 4. Dataloader

Once the dataset is prepared, the dataloader for using the dataset is present in **`dataloader.py`**. This handles loading, transforming, and batching the dataset.

For a sample output of input shapes, run the following
```bash
python dataloader.py
```

You will get a sample image shapes, labels when you run this script.

---