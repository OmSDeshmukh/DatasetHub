# Dataset: ImageNet-1k

Welcome to the **ImageNet-1k** dataset folder! This directory contains all the necessary resources to download, prepare, and load the dataset for your machine learning tasks.

---

## 🚀 Getting Started

To get started with **ImageNet-1k** dataset, follow these steps:


### 1. Requirements

You might need to download some python packages for running the `dataprep.py` and `dataloader.py`. Run the following commands to download required python packages.

```bash
pip install torch torchvision torchaudio pillow numpy tqdm multiprocess
```

### 2. Download the Dataset

You can download the dataset from the official source by creating an official login on the website and agreeing to its terms and conditions of use.
- [Download **ImageNet-1k** Dataset](https://www.image-net.org/download.php)

- Here I provide another way to download the dataset from a repository on HuggingFace [here](https://huggingface.co/datasets/ILSVRC/imagenet-1k), 


#### Instructions:
When using the HuggingFace method, the code for both the download and data preparation is present in the `dataprep.py` file itself.

#### Note:
You may need to agree to the dataset's terms of use before downloading the files.


### 3. Prepare the Dataset

Run the file **`dataprep.py`** to download and prepare the data for use. This script handles tasks like downloading the archives, organizing files, and converting the dataset into the proper format for machine learning.

Run the following command to begin preparing the dataset:
```bash
python dataprep.py
```

Here dataprep mainly involves downloads the files from the Hugging Face servers. Then it creates the `train`, `validation` and `test` folders which contains folders named using class-ids from 0 to 999.

#### Note
`test` only contains a single folder named `-1` which contains the images for testing. You have to use your model for inference on all the images in this folder and submit it to the official website for results in a desired format. For more information, refer to the official website [here](https://www.image-net.org/download.php)

### 4. Dataloader

Once the dataset is prepared, the dataloader for using the dataset is present in **`dataloader.py`**. This handles loading, transforming, and batching the dataset.

For a sample output of input shapes, run the following
```bash
python dataloader.py
```

You will get a sample image shapes, labels when you run this script

---