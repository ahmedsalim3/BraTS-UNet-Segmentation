# BraTS-UNet-Segmentation

- Implementing a a custom deep learning model based on the U-Net architecture, enhanced with features like recurrent atrous spatial pyramid pooling, attention mechanisms, and RRCNN blocks for MRI brain tumor segmentation

- Processed dataset is available [here](https://www.kaggle.com/datasets/ahvshim/mri-brats2019-training-and-validation-splits/data)

## Repo's directory structure

The directory structure below shows the nature of files/directories used in this repo.

```sh
BRATS-Image-Segmentation
│
├── README.md
│
├── data              
│   ├── BraTS2019              # Main data
│   ├── interim                # Intermediate data that has been processed
│   └── processed
│       ├── train              # Processed training data
│       └── valid              # Processed validation data
│
├── models
│   └── Model.png              # Model architecture visualization
│
├── notebooks
│   ├── Brats_DL.ipynb         # Notebook for model training
│   └── Brats_Prep.ipynb       # Notebook for data preprocessing
│
└── src
    ├── preprocessing
    │   ├── N4biasfieldcor.py  # Bias field correction
    │   ├── image_cropper.py   # Crop 3D images
    │   ├── image_processor.py # Process 3D images and masks
    │   ├── create_folds.py    # Data creation and folding
    │   ├── utils.py           # Preprocessing utilities
    │   └── __init__.py
    ├── Modules.py             # DLUNetModel and ImageVisualizer classes
    ├── __init__.py
    └── utils.py               # General utilities

```
