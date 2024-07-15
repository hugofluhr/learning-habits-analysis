#!/bin/sh

data_path="/home/ubuntu/data/bids_dataset"

cd $data_path

for subject in sub-*; do
    anat_path="$data_path/$subject/ses-1/anat"
    echo $anat_path
    T1W="$anat_path/${subject}_ses-1_run-1_T1w.nii"
    T1W_bis="$anat_path/${subject}_ses-1_run-2_T1w.nii"
    
    if [ -f "$T1W" ]; then
        echo "Processing $T1W"
        pydeface "$T1W"
    else
        echo "File $T1W does not exist."
    fi

    # check if second anatomical scan exists
    if [ -f "$T1W_bis" ]; then
        echo "Processing $T1W_bis"
        pydeface "$T1W_bis"
    fi

done
