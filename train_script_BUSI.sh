#!/bin/bash

# Define the path to the dataset and the base output directory
DATA_PATH="dataset/BUSI/"
BASE_OUTPUT_DIR="checkpoints/results_BUSI"

# Define the total number of epochs for training
EPOCHS=100
KFOLDS=5

# Define all model architectures to train
declare -a architectures=("resnet50" "vgg16" "vit-ti16" "vit-s16" "vit-s32" "vit-b16" "vit-b32" "vim-s" "vssm-ti" "vssm-s" "vssm-b")

# Loop through each architecture and run the training
for arch in "${architectures[@]}"
do
    echo "--------------------------------------------------------------------------------"
    echo "Starting training for $arch with $KFOLDS folds."
    echo "--------------------------------------------------------------------------------"

    # Define the output directory for this architecture
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$arch/"

    # Run the training command with k-folds parameter
    python3 main.py --epochs $EPOCHS --data-path $DATA_PATH --output_dir $OUTPUT_DIR --arch $arch --k-folds $KFOLDS

    echo "--------------------------------------------------------------------------------"
    echo "Training completed for $arch"
    echo "--------------------------------------------------------------------------------"
done

echo "All training processes are completed."