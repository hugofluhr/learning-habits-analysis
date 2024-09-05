#!/bin/bash
for s in {2..73}; do
    python scripts/correct_fmap_data.py --source_folder /mnt/data/scanner_raw/ --bids_folder /mnt/data/bids_dataset/ -s $s
    echo "Completed processing subject $s"
done