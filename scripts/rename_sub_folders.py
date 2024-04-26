import os
import csv
import argparse

def create_lookup_table(directory):
    lookup_table = {}
    subdirectories = [subdir for subdir in os.listdir(directory) if subdir.startswith('SNS_MRI_LH_')]
    subdirectories.sort(key=lambda x: os.path.getctime(os.path.join(directory, x)))
    for index, subdirectory in enumerate(subdirectories):
        new_name = f"sub-{index+1:02}"
        lookup_table[subdirectory] = new_name

    return lookup_table

def rename_sub_folders(directory, lookup_table):
    for old_name, new_name in lookup_table.items():
        os.rename(os.path.join(directory, old_name), os.path.join(directory, new_name))

def save_lookup_table(directory, lookup_table):
    lookup_table_file = os.path.join(directory, 'subject_lookup_table.csv')
    with open(lookup_table_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Old Name", "New Name"])
        for old_name, new_name in lookup_table.items():
            writer.writerow([old_name, new_name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename sub-folders in a directory")
    parser.add_argument("-d", "--directory", help="Path to the raw data directory", required=True, type=str)
    args = parser.parse_args()

    directory = args.directory
    lookup_table = create_lookup_table(directory)
    rename_sub_folders(directory, lookup_table)
    save_lookup_table(directory, lookup_table)

    print("Lookup table:")
    for old_name, new_name in lookup_table.items():
        print(f"{old_name} -> {new_name}")