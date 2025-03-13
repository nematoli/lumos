#!/bin/bash

# Download, Unzip, and Remove zip
if [ "$1" = "calvin" ]
then
    echo "Downloading calvin task_D_D ..."
    wget http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip task_D_D.zip && rm task_D_D.zip
    mv task_A_A task_D_D
    echo "saved folder: task_D_D"
elif [ "$1" = "lumos_dataset" ]; then
    echo "Downloading LUMOS dataset ..."
    wget http://lumos.cs.uni-freiburg.de/datasets/lumos_dataset.zip
    unzip lumos_dataset.zip && rm lumos_dataset.zip
    echo "Saved folder: lumos_dataset"
else
    echo "Failed: Usage download_data.sh calvin | XXX "
    exit 1
fi
