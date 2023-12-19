#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

DATA_DIR="../data"
if [ ! -d "$DATA_DIR" ]; then
    mkdir "$DATA_DIR"
fi


# Download the dataset
kaggle datasets download -d  muhammadbinimran/housing-price-prediction-data -p "$DATA_DIR"

# Change to the data directory and unzip the dataset
cd "$DATA_DIR"
unzip '*.zip'
rm *.zip