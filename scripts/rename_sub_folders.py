import os
import csv
import argparse
import datetime

def update_lookup_table(directory, lookup_table):
    print('\nNew participants: ')
    nb_renamed_participants = len(lookup_table)
    new_participants = [subdir for subdir in os.listdir(directory) if subdir.startswith('SNS_MRI_LH_')]
    new_participants.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))

    lookup_table_file = os.path.join(directory, 'subjects_lookup_table.csv')
    with open(lookup_table_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for index, old_name in enumerate(new_participants):
            new_name = f"SNS_MRI_LH_sub-{index+1+nb_renamed_participants:02}"
            lookup_table[old_name] = new_name
            print(f"{old_name} -> {new_name}")
            # somehow creation time gives the date of copying the data, mtime gives the date of creation (on the scanner computer)
            creation_date = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(directory, old_name))).strftime("%Y-%m-%d %H:%M:%S")
            renaming_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([old_name, new_name, creation_date, renaming_date])
    return lookup_table

def load_lookup_table(lookup_table_file):
    print('\nAlready renamed: ')
    lookup_table = {}
    with open(lookup_table_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            old_name, new_name = row[:2]
            lookup_table[old_name] = new_name
            print(f"{old_name} -> {new_name}")
    return lookup_table

def rename_sub_folders(directory, lookup_table):
    for old_name, new_name in lookup_table.items():
        if os.path.exists(os.path.join(directory, old_name)):
            os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))

def save_lookup_table(directory, lookup_table):
    lookup_table_file = os.path.join(directory, 'subjects_lookup_table.csv')
    with open(lookup_table_file, 'a', newline='') as file:
        writer = csv.writer(file)
        for old_name, new_name in lookup_table.items():
            writer.writerow([old_name, new_name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename sub-folders in a directory")
    parser.add_argument("-d", "--directory", help="Path to the raw data directory", required=True, type=str)
    args = parser.parse_args()
    directory = args.directory

    lookup_table_file = os.path.join(args.directory, 'subjects_lookup_table.csv')
    if os.path.exists(lookup_table_file):
        print("Lookup table already exists. Updating it...")
        lookup_table = load_lookup_table(lookup_table_file)
    else:
        print("Creating new lookup table...")
        lookup_table = {}

    lookup_table = update_lookup_table(directory, lookup_table)
    rename_sub_folders(directory, lookup_table)

    print("\nDone!")