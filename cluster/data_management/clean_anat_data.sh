#!/bin/bash

# handle subjects with 2 T1 images
# for i in 10 22 33 57 63; do
#   rm "/home/hfluhr/shares-hare/ds-learning-habits/sub-${i}/ses-1/anat/sub-${i}_ses-1_run-1_T1w_defaced.nii"
#   mv "/home/hfluhr/shares-hare/ds-learning-habits/sub-${i}/ses-1/anat/sub-${i}_ses-1_run-2_T1w_defaced.nii" "/home/hfluhr/shares-hare/ds-learning-habits/sub-${i}/ses-1/anat/sub-${i}_ses-1_run-1_T1w_defaced.nii"
# done

for file in /home/hfluhr/shares-hare/ds-learning-habits/sub-*/ses-1/anat/*T1w_defaced.nii; do
  # Extract the directory name
  dir=$(dirname "$file")
  # Extract the base name of the file (without the directory part)
  base=$(basename "$file" .nii)
  # Construct the new file name
  new_file="${dir}/${base/_defaced/}.nii"
  # Rename the file
  mv "$file" "$new_file"
  #echo $file
  #echo $new_file
done