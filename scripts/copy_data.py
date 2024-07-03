import os
import shutil
import argparse
import re
import csv

def copy_directories(source_dir, target_dir, prefix, dry_run=0):
    regex_pattern = re.compile(f'^{prefix}(.{{6}})_')
    
    copied_subjects = []
    # first check existing folders in target_dir
    for dirname in os.listdir(target_dir):
        match = regex_pattern.match(dirname)
        if match:
            copied_subjects.append(match.group(1))
    # then check if there already is a subject lookup table
    lut_path = os.path.join(target_dir, 'subjects_lookup_table.csv')
    if os.path.isfile(lut_path):
        with open(lut_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                match = regex_pattern.match(row[0])
                if match:
                    copied_subjects.append(match.group(1))

    for root, dirs, _ in os.walk(source_dir):
        for dir_name in dirs:
            match = regex_pattern.match(dir_name)
            if match:
                subject_id = match.group(1)
                if subject_id not in copied_subjects:
                    source_path = os.path.join(root, dir_name)
                    target_path = os.path.join(target_dir, dir_name)
                    if dry_run:
                        print('Dry run: would copy', source_path, 'to', target_path)
                    else:
                        print('Copying data from', dir_name)
                        shutil.copytree(source_path, target_path)
                        copied_subjects.append(subject_id)
                else:
                    print('Skipping', dir_name, 'since it has already been copied')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy directories based on a pattern")
    parser.add_argument("-s", "--sourcedir", help="Path to the source directory")
    parser.add_argument("-t", "--targetdir", help="Path to the target directory")
    parser.add_argument('-n', dest='dry_run',default=0, help="dry run", type=int)
    args = parser.parse_args()

    subject_prefix = "SNS_MRI_LH_"
    
    if not os.path.exists(args.targetdir) and not args.dry_run:
        os.makedirs(args.targetdir)
    
    copy_directories(args.sourcedir, args.targetdir, subject_prefix, args.dry_run)

    print('Done!')