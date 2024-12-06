
# Contributing to DatasetHub

I am excited that you’re interested in contributing to **DatasetHub**! By sharing your dataset setup, you can help the machine learning community save time and effort.

---

## 🛠 How to Contribute

To contribute a dataset, follow these steps:

### 1. Fork the Repository
- Click the **Fork** button on the top right of this repository page to create a copy of the repository in your GitHub account.

### 2. Clone the Forked Repository
- Clone your forked repository to your local machine using the following command:
  ```bash
  git clone https://github.com/<your-username>/DatasetHub.git
  ```

### 3. Create a New Branch
- Navigate to the repository folder on your machine:
  ```bash
  cd DatasetHub
  ```
- Create and switch to a new branch for your contribution:
  ```bash
  git checkout -b add-dataset-name
  ```

### 4. Add Your Dataset Setup
- Locate the appropriate **modality** and **task** folders for your dataset. If they don’t exist, create them.
- Inside the correct folder, create a new directory named after your dataset.
- Add the following three files mentioned below to your dataset folder.
- You can also have additional files as per the requirement of the dataset, but these 3 files are mandatory.


#### a) `README.md` (Required)
- Provide detailed download and setup instructions for the dataset.
- Include links to the official dataset page.
- Refer the other datasets `README.MD` files for the basic structure. The sections present in them should be followed. Nevertheless, you can add your own necessary sections if required.

#### b) `dataprep.py` (Required)
- Write a script to prepare the dataset (e.g., unzip, preprocess, or reorganize files).

#### c) `dataloader.py` (Required)
- Write a script to load the dataset into your machine learning pipeline.

### 5. Commit Your Changes
- Stage your changes:
  ```bash
  git add .
  ```
- Commit your changes with a meaningful message:
  ```bash
  git commit -m "Add setup for [Dataset Name]"
  ```

### 6. Push Your Changes
- Push your branch to your forked repository:
  ```bash
  git push origin add-dataset-name
  ```

### 7. Create a Pull Request
- Go to the original **DatasetHub** repository on GitHub.
- Click the **Pull Request** tab, then click **New Pull Request**.
- Select your branch and submit a pull request with a detailed description of your contribution.

---

## ✅ Contribution Checklist
Before submitting your pull request, ensure that:
- [ ] You have created the dataset folder in the correct location.
- [ ] Your dataset folder contains `README.md`, `dataprep.py`, and `dataloader.py`.
- [ ] Your scripts have been tested locally.
- [ ] Your `README.md` includes clear setup and usage instructions.

---

## 🔍 Review and Approval Process
1. Once you create a pull request, it will be reviewed by the maintainers.
2. If changes are requested, update your branch and push the changes.
3. Once approved, your changes will be merged into the main branch.

---

Thank you for your contribution! If you have any questions or face issues while contributing, feel free to [open an issue](https://github.com/OmSDeshmukh/DatasetHub/issues).
