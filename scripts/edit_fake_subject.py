# usage:
# python /home/hfluhr/repos/learning-habits-analysis/scripts/edit_fake_subject.py /home/hfluhr/data/learninghabits/sdc_test/sub-91 sub-01 sub-91 --echo_spacing 0.05 --readout_time 1
# old values:
# "TotalReadoutTime": 0.046933141586711837
# "EffectiveEchoSpacing": 0.0005940904

import os
import argparse
import json

def rename_bids_files(root_dir, old_sub_id, new_sub_id):
    """
    Rename all files and directories in a BIDS dataset folder to reflect the new subject ID.
    
    Parameters:
        root_dir (str): Path to the root directory of the BIDS dataset.
        old_sub_id (str): The original subject ID (e.g., "sub-01").
        new_sub_id (str): The new subject ID (e.g., "sub-91").
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for filename in filenames:
            if old_sub_id in filename:
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace(old_sub_id, new_sub_id)
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")
        for dirname in dirnames:
            if old_sub_id in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace(old_sub_id, new_sub_id)
                new_path = os.path.join(dirpath, new_dirname)
                os.rename(old_path, new_path)
                print(f"Renamed directory: {old_path} -> {new_path}")

def update_json_parameters(root_dir, echo_spacing, readout_time):
    """
    Update EffectiveEchoSpacing and TotalReadoutTime in all func JSON sidecar files.
    
    Parameters:
        root_dir (str): Path to the root directory of the BIDS dataset.
        echo_spacing (float): New value for EffectiveEchoSpacing.
        readout_time (float): New value for TotalReadoutTime.
    """
    func_dir = os.path.join(root_dir, "ses-1/func")
    if not os.path.exists(func_dir):
        print(f"No 'func' directory found in {root_dir}. Skipping JSON update.")
        return
    
    for dirpath, _, filenames in os.walk(func_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                try:
                    with open(json_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                    
                    data["EffectiveEchoSpacing"] = echo_spacing
                    data["TotalReadoutTime"] = readout_time

                    with open(json_path, "w", encoding="utf-8") as file:
                        json.dump(data, file, indent=4)
                    
                    print(f"Updated JSON file: {json_path}")
                except Exception as e:
                    print(f"Error updating {json_path}: {e}")

def update_fmap_intended_for(root_dir, old_sub_id, new_sub_id):
    """
    Update the IntendedFor field in all fmap JSON sidecar files to reflect the new subject ID.
    
    Parameters:
        root_dir (str): Path to the root directory of the BIDS dataset.
        old_sub_id (str): The original subject ID (e.g., "sub-01").
        new_sub_id (str): The new subject ID (e.g., "sub-91").
    """
    fmap_dir = os.path.join(root_dir, "ses-1/fmap")
    if not os.path.exists(fmap_dir):
        print(f"No 'fmap' directory found in {root_dir}. Skipping IntendedFor update.")
        return
    
    for dirpath, _, filenames in os.walk(fmap_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                try:
                    with open(json_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                    
                    if "IntendedFor" in data and isinstance(data["IntendedFor"], list):
                        updated_intended_for = [
                            entry.replace(old_sub_id, new_sub_id)
                            for entry in data["IntendedFor"]
                        ]
                        data["IntendedFor"] = updated_intended_for

                    with open(json_path, "w", encoding="utf-8") as file:
                        json.dump(data, file, indent=4)
                    
                    print(f"Updated IntendedFor in JSON file: {json_path}")
                except Exception as e:
                    print(f"Error updating IntendedFor in {json_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rename files/directories and update JSON parameters in a BIDS dataset."
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to the root directory of the BIDS dataset (e.g., '/path/to/BIDS/dataset/sub-91')."
    )
    parser.add_argument(
        "old_sub_id",
        type=str,
        help="The original subject ID to be replaced (e.g., 'sub-01')."
    )
    parser.add_argument(
        "new_sub_id",
        type=str,
        help="The new subject ID to replace the old one (e.g., 'sub-91')."
    )
    parser.add_argument(
        "--echo_spacing",
        type=float,
        default=None,
        help="New value for EffectiveEchoSpacing in func JSON files."
    )
    parser.add_argument(
        "--readout_time",
        type=float,
        default=None,
        help="New value for TotalReadoutTime in func JSON files."
    )

    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"Error: The specified path '{args.root_dir}' does not exist.")
    else:
        rename_bids_files(args.root_dir, args.old_sub_id, args.new_sub_id)
        
        if args.echo_spacing is not None and args.readout_time is not None:
            update_json_parameters(args.root_dir, args.echo_spacing, args.readout_time)
        else:
            print("Skipping JSON parameter update as no values were provided.")
        
        update_fmap_intended_for(args.root_dir, args.old_sub_id, args.new_sub_id)