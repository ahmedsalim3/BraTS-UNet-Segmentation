#!/bin/bash

# Download data if not exsist
if [ ! -d "data/BraTS2019/HGG" ] && [ ! -d "data/BraTS2019/LGG" ]; then
  echo "Data not found, downloading..."
  bash scripts/datasets/download_raw_dataset.sh
else
  echo "Data already exists, skipping download."
fi

python3 -m src.process_dataset