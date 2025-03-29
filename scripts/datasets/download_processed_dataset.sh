#!/bin/bash

# This script will download the processed BraTS2019 dataset from Kaggle:
# https://www.kaggle.com/datasets/ahvshim/mri-brats2019-training-and-validation-splits,
# and save it in the 'data/processed' folder.

# Ensure you have the Kaggle API key as described here:
# https://github.com/Kaggle/kaggle-api

DATASET_NAME="ahvshim/mri-brats2019-training-and-validation-splits"
KAGGLE_CREDS_PATH="/home/ahmedsalim/.kaggle/kaggle.json" # Replace this with the actual path to your kaggle.json file

if [ ! -f "$KAGGLE_CREDS_PATH" ]; then
  echo "Kaggle credentials file (kaggle.json) not found!"
  exit 1
fi

echo "Installing Kaggle API..."
pip install kaggle --quiet

export KAGGLE_CONFIG_DIR=$(dirname "$KAGGLE_CREDS_PATH")

mkdir -p data
echo "Downloading dataset..."
kaggle datasets download -d $DATASET_NAME -p .

ZIP_FILE=$(ls mri-brats2019-training-and-validation-splits.zip)
echo "Download complete, extracting the ZIP file..."
unzip -q "$ZIP_FILE" -d .

rm -rf "$ZIP_FILE"
mv "BraTS2019" "data/processed"
echo "Download and extraction complete!"