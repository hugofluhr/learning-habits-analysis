#!/bin/bash

# Define the paths
LOCAL_BASE_DIR="/home/ubuntu/data/bids_dataset/"
DEST_DIR="/mnt/econ_department/projects/2024/nebe_fluhr_timokhov_tobler_learning_habits/data/bids_dataset/"

# Run rsync to copy only the renamed files
sudo rsync -avz --copy-links "$LOCAL_BASE_DIR" "$DEST_DIR"
