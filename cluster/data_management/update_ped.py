import os
import json
import argparse
import glob

def update_ped(json_path, ped='j', dry_run=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'PhaseEncodingDirection' in data:
        original_ped = data['PhaseEncodingDirection']
        
        new_ped = ped
        if not dry_run:
            data['PhaseEncodingDirection'] = new_ped
            data.pop('IntendedFor', None)
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        
        return original_ped, new_ped
    else:
        return None, None

def process_files(files, ped = 'j', dry_run=False):
    for json_path in files:
        original, updated = update_ped(json_path, ped=ped, dry_run=dry_run)
        if original and updated:
            print(f"Processed {json_path}")
            print(f"Original PhaseEncodingDirection: {original}")
            print(f"Updated PhaseEncodingDirection: {updated}")
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update PhaseEncodingDirection BIDS dataset JSON files within specified directories.")
    parser.add_argument("directory", type=str, help="Base directory containing BIDS dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files.")

    args = parser.parse_args()

    # Use glob to find all the JSON files based on the pattern
    pattern = os.path.join(args.directory, 'sub-*/ses-1/func/*bold.json')
    files = glob.glob(pattern)
    files.sort()

    process_files(files, ped='j', dry_run=args.dry_run)
