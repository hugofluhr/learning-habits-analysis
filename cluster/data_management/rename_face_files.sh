#!/bin/bash

# Iterate over each file matching the pattern
for file in ~/shares-hare/ds-learning-habits/sub-*/ses-1/anat/*T1w.nii; do
  # Extract the directory name
  dir=$(dirname "$file")
  # Extract the base name of the file (without the directory part)
  base=$(basename "$file" .nii)
  # Construct the new file name
  new_file="${dir}/${base}_with_face.nii"
  # Rename the file
  mv "$file" "$new_file"
  # echo $new_file
done

