#!/bin/bash
for s in {1..73}; do
    python scripts/convert_to_bids.py --source_folder /mnt/data/scanner_raw/ --bids_folder /mnt/data/bids_dataset/ -s $s
    echo "Completed processing subject $s"
done
