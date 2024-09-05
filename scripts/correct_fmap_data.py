'''
adapted from scripts by Gilles de Hollander and Ella Casimiro.
usage:
    python scripts/convert_to_bids.py --source_folder ~/data/scanner_raw/ --bids_folder ~/data/bids/ds-learninghabits/ -s 1
'''
import re
import os
import os.path as op
import shutil
import argparse
import glob
import json

# Lookup table for runs
runs_lookup = {
    1: 'learning',
    2: 'learning',
    3: 'test'
}

def main(subject, source_folder, bids_folder='/data'):
    # set session to 1 as we only have a single session
    session = 1

    file_names_lookup = {}

    try:
        subject = int(subject)
        subject = f'{subject:02d}'
    except ValueError:
        pass

    ## fix this to record subject ID at this point
    sourcedata_root = [os.path.join(root, dir) for root, dirs, files in os.walk(source_folder) for dir in dirs if dir == f'SNS_MRI_LH_sub-{subject:02}']
    try :
        sourcedata_root = sourcedata_root[0]
    except IndexError:
        print(f"folder not found for subject {subject}")
        return
    #sourcedata_root = op.join(bids_folder, 'sourcedata', 'mri', f'sub-{subject}', f'ses-{session}')

    # # # *** ANATOMICAL DATA ***
    t1w = glob.glob(op.join(sourcedata_root, '*_t1w*.nii'))
    print(t1w)
    assert (len(t1w) != 0), "No T1w {t1w}"
    anat_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'anat')
    if not op.exists(anat_dir):
        os.makedirs(anat_dir)
    # there should be only one T1w
    for run0, t in enumerate(t1w):
        if not op.exists(op.join(anat_dir, f'sub-{subject}_ses-{session}_run-{run0+1}_T1w.nii')):
            raise Exception("missing file")
        #shutil.copy(t, op.join(anat_dir, f'sub-{subject}_ses-{session}_run-{run0+1}_T1w.nii'))
        file_names_lookup.update({t: op.join(anat_dir, f'sub-{subject}_ses-{session}_run-{run0+1}_T1w.nii')})

    # # # *** FUNCTIONAL DATA ***
    func_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'func')
    if not op.exists(func_dir):
        os.makedirs(func_dir)

    # *** functional data ***
    # loading the json template for bold run
    with open(op.abspath('./utils/bold_template.json'), 'r') as f:
        json_template = json.load(f)

    # Finding the bold runs
    # this one from Gilles and Ella didn't work: '.*run(?P<run>[0-9]+).*'
    reg = re.compile('.*_run_(?P<run>\d+)s.*$')
    funcs = glob.glob(op.join(sourcedata_root, '*run*.nii'))
    runs = [int(reg.match(fn).group('run')) for fn in funcs]
    for run, fn in zip(runs, funcs):
        task = runs_lookup[run]
        target_file_name = f'sub-{subject}_ses-{session}_task-{task}_run-{run}_bold.nii'
        json_template['task'] = task
        if not op.exists(op.join(func_dir, target_file_name)):
            raise Exception("missing file")
        #shutil.copy(fn, op.join(func_dir, target_file_name))
        file_names_lookup.update({fn: op.join(func_dir, target_file_name)})

        #json_sidecar = json_template
        #with open(op.join(func_dir, target_file_name.replace('.nii','.json')), 'w') as f:
        #    json.dump(json_sidecar, f)

    # *** physio logfiles ***
    physiologs = glob.glob(op.join(sourcedata_root, '*run*scanphyslog*.log'))
    runs = [int(reg.match(fn).group(1)) for fn in physiologs]
    for run, fn in zip(runs, physiologs):
        task = runs_lookup[run]
        target_file_name = f'sub-{subject}_ses-{session}_task-{task}_run-{run}_physio.log'
        if not op.exists(op.join(func_dir, target_file_name)):
            raise Exception("missing file")
        #shutil.copy(fn, op.join(func_dir, target_file_name))
        file_names_lookup.update({fn: op.join(func_dir, target_file_name)})

    # # # *** FIELDMAPS ***
    fmap_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'fmap')
    if not op.exists(fmap_dir):
        os.makedirs(fmap_dir)

    b0_fieldmaps_mag1 = glob.glob(op.join(sourcedata_root, '*bo_fieldmap_v01_ec1*_typ0.nii'))
    b0_fieldmaps_mag1.sort()
    b0_fieldmaps_mag2 = glob.glob(op.join(sourcedata_root, '*bo_fieldmap_v01_ec2*_typ0.nii'))
    b0_fieldmaps_mag2.sort()
    b0_fieldmaps_phase1 = glob.glob(op.join(sourcedata_root, '*bo_fieldmap_v01_ec1*_typ3.nii'))
    b0_fieldmaps_phase1.sort()
    b0_fieldmaps_phase2 = glob.glob(op.join(sourcedata_root, '*bo_fieldmap_v01_ec2*_typ3.nii'))
    b0_fieldmaps_phase2.sort()

    physiologs = glob.glob(op.join(sourcedata_root, '*fieldmap*scanphyslog*.log'))
    physiologs.sort()

    runs = [i+1 for i in range(len(b0_fieldmaps_mag1))]

    # *** physio logfiles ***
    for run, fn in zip(runs, physiologs):
        task = runs_lookup[run]
        target_file_name = f'sub-{subject}_ses-{session}_task-{task}_run-{run}_physio.log'
        shutil.copy(fn, op.join(fmap_dir, target_file_name))
        file_names_lookup.update({fn: op.join(fmap_dir, target_file_name)})

    # *** fieldmaps ***
    for run, fn in zip(runs, b0_fieldmaps_mag1):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_magnitude1.nii'))
        file_names_lookup.update({fn: op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_magnitude1.nii')})
        json_sidecar = dict()

        task = runs_lookup[run]
        json_sidecar[
                'IntendedFor'] = [f'bids::sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-{task}_run-{run}_bold.nii']
        
        with open(op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_magnitude1.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.0046
            json.dump(json_sidecar, f)

        with open(op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_magnitude2.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.00694
            json.dump(json_sidecar, f)

        with open(op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_phase1.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.0046
            json.dump(json_sidecar, f)

        with open(op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_phase2.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.00694
            json.dump(json_sidecar, f)

    for run, fn in zip(runs, b0_fieldmaps_mag2):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_magnitude2.nii'))
        file_names_lookup.update({fn: op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_magnitude2.nii')})

    for run, fn in zip(runs, b0_fieldmaps_phase1):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_phase1.nii'))
        file_names_lookup.update({fn: op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_phase1.nii')})

    for run, fn in zip(runs, b0_fieldmaps_phase2):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_phase2.nii'))
        file_names_lookup.update({fn: op.join(fmap_dir, f'sub-{subject}_ses-{session}_run-{run}_phase2.nii')})
    
    with open(op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'file_names_lookup.json'), 'w') as f:
        json.dump(file_names_lookup, f, separators=('\n','\n'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--subject', default=1, type=str)
    parser.add_argument('--source_folder', default='/Users/hugofluhr/phd_local/data/LearningHabits/scanner_raw')
    parser.add_argument('--bids_folder', default='/Users/hugofluhr/phd_local/data/LearningHabits/bids_formatted/ds-learninghabits')
    args = parser.parse_args()

    main(args.subject, source_folder = args.source_folder,bids_folder=args.bids_folder)
