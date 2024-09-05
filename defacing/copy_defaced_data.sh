#!/bin/bash

# Remote server details
REMOTE_USER="hfluhr"
REMOTE_HOST="cluster.s3it.uzh.ch"
REMOTE_DIR="shares-hare/ds-learning-habits/"

# Define the local base directory
LOCAL_BASE_DIR="/home/ubuntu/data/bids_dataset/"

# Run rsync to copy only the renamed files
rsync -avz --include '*/' --include '*T1w_defaced.nii' --exclude '*' "$LOCAL_BASE_DIR" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
