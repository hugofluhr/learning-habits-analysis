#!/bin/bash

# Remote server details
REMOTE_USER="hfluhr"
REMOTE_HOST="cluster.s3it.uzh.ch"
REMOTE_DIR="shares-hare/ds-learning-habits/derivatives/"

# Define the local base directory
LOCAL_BASE_DIR="/home/ubuntu/data/bids_dataset/derivatives/"

# Run rsync to copy only the renamed files
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_BASE_DIR"