import os
import json
import argparse
import glob

def update_intended_for(json_path, dry_run=False):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'IntendedFor' in data:
        original_intended_for = data['IntendedFor']
        new_intended_for = []
        
        for item in original_intended_for:
            if item.startswith("bids::"):
                relative_path = item.split("bids::")[1]
                # Extract the session and subsequent components
                parts = relative_path.split('/')
                # Create the new relative path
                relative_path = os.path.join(parts[1], *parts[2:])
                new_intended_for.append(relative_path)
            else:
                new_intended_for.append(item)
        
        if not dry_run:
            data['IntendedFor'] = new_intended_for
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
        
        return original_intended_for, new_intended_for
    else:
        return None, None

def process_files(files, dry_run=False):
    for json_path in files:
        original, updated = update_intended_for(json_path, dry_run=dry_run)
        if original and updated:
            print(f"Processed {json_path}")
            print(f"Original IntendedFor: {original}")
            print(f"Updated IntendedFor: {updated}")
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update IntendedFor field in BIDS dataset JSON files within specified directories.")
    parser.add_argument("directory", type=str, help="Base directory containing BIDS dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without modifying files.")

    args = parser.parse_args()

    # Use glob to find all the JSON files based on the pattern
    pattern = os.path.join(args.directory, 'sub-*/ses-1/fmap/*.json')
    files = glob.glob(pattern)
    files.sort()

    process_files(files, dry_run=args.dry_run)
