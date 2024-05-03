'''
adapted from scripts by Gilles de Hollander and Ella Casimiro.
'''
import re
import os
import os.path as op
import shutil
import argparse
import glob
from nilearn import image
import numpy as np
import json

def main(subject, bids_folder='/data'):
    # set session to 1 as we only have a single session
    session = 1

    try:
        subject = int(subject)
        subject = f'{subject:02d}'
    except ValueError:
        pass
    
    # when it works: 

    # dev folder
    sourcedata_folder = '/Users/hugofluhr/phd_local/data/LearningHabits/scanner_raw'

    ## fix this to record subject ID at this point
    sourcedata_root = [os.path.join(root, dir) for root, dirs, files in os.walk(sourcedata_folder) for dir in dirs if dir == f'SNS_MRI_LH_sub-{subject:02}']
    try :
        sourcedata_root = sourcedata_root[0]
    except IndexError:
        print(f"folder not found for {subject}")
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
        shutil.copy(t, op.join(anat_dir, f'sub-{subject}_ses-{session}_run-{run0+1}_T1w.nii'))

    # # # *** FUNCTIONAL DATA ***
    func_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'func')
    if not op.exists(func_dir):
        os.makedirs(func_dir)

    # *** functional data ***
    # loading the json template for bold run
    with open(op.abspath('./utils/bold_training_template.json'), 'r') as f:
        json_template = json.load(f)

    # Finding the bold runs
    # this one from Gilles and Ella didn't work: '.*run(?P<run>[0-9]+).*'
    reg = re.compile('.*_run_(?P<run>\d+)s.*$')
    funcs = glob.glob(op.join(sourcedata_root, '*run*.nii'))
    runs = [int(reg.match(fn).group('run')) for fn in funcs]
    for run, fn in zip(runs, funcs):
        if run == 3: # test phase
            target_file_name = f'sub-{subject}_ses-{session}_task-test_run-{run}_bold.nii'
            json_template['task'] = 'test'
        else: # learning blocks 1 and 2
            target_file_name = f'sub-{subject}_ses-{session}_task-learning_run-{run}_bold.nii'
            json_template['task'] = 'learning'
        shutil.copy(fn, op.join(func_dir, target_file_name))

        json_sidecar = json_template
        with open(op.join(func_dir, target_file_name.replace('.nii','.json')), 'w') as f:
            json.dump(json_sidecar, f)

    # *** physio logfiles ***
    physiologs = glob.glob(op.join(sourcedata_root, '*run*scanphyslog*.log'))
    runs = [int(reg.match(fn).group(1)) for fn in physiologs]
    for run, fn in zip(runs, physiologs):
        if run == 3: # test phase
            target_file_name = f'sub-{subject}_ses-{session}_task-test_run-{run}_physio.log'
        else: # learning blocks 1 and 2
            target_file_name = f'sub-{subject}_ses-{session}_task-learning_run-{run}_physio.log'
        shutil.copy(fn, op.join(func_dir, target_file_name))

    # # # *** FIELDMAPS ***
    fmap_dir = op.join(bids_folder, f'sub-{subject}', f'ses-{session}', 'fmap')
    if not op.exists(fmap_dir):
        os.makedirs(fmap_dir)

    b0_fieldmaps_mag1 = glob.glob(op.join(sourcedata_root, '*bo_fieldmap_v01_ec1*_typ0.nii'))
    b0_fieldmaps_mag2 = glob.glob(op.join(sourcedata_root, '*bo_fieldmap_v01_ec2*_typ0.nii'))
    b0_fieldmaps_phase1 = glob.glob(op.join(sourcedata_root, '*bo_fieldmap_v01_ec1*_typ3.nii'))
    b0_fieldmaps_phase2 = glob.glob(op.join(sourcedata_root, '*bo_fieldmap_v01_ec2*_typ3.nii'))

    physiologs = glob.glob(op.join(sourcedata_root, '*fieldmap*scanphyslog*.log'))


    runs = [i+1 for i in range(len(b0_fieldmaps_mag1))]

    for run, fn in zip(runs, physiologs):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_physio.log'))

    for run, fn in zip(runs, b0_fieldmaps_mag1):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_magnitude1.nii'))
        json_sidecar = dict()
        if run == 1:
            json_sidecar[
                'IntendedFor'] = [f'func/sub-{subject:02d}_task-learn_run-1_bold.nii']
        elif run == 2:
            json_sidecar[
                'IntendedFor'] = [f'func/sub-{subject:02d}_task-learn_run-2_bold.nii',
                                  f'func/sub-{subject:02d}_task-learn_run-3_bold.nii']
        elif run == 3:
            json_sidecar[
                'IntendedFor'] = [f'func/sub-{subject:02d}_task-learn_run-4_bold.nii',
                                  f'func/sub-{subject:02d}_task-learn_run-5_bold.nii']
        elif run == 4:
            json_sidecar[
                'IntendedFor'] = [f'func/sub-{subject:02d}_task-learn_run-6_bold.nii']
        else:
            raise Exception("Run not between 1 and 4")


        with open(op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_magnitude1.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.0046
            json.dump(json_sidecar, f)

        with open(op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_magnitude2.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.006940000000000001
            json.dump(json_sidecar, f)

        with open(op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_phase1.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.0046
            json.dump(json_sidecar, f)

        with open(op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_phase2.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.006940000000000001
            json.dump(json_sidecar, f)

    for run, fn in zip(runs, b0_fieldmaps_mag2):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_magnitude2.nii'))

    for run, fn in zip(runs, b0_fieldmaps_phase1):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_phase1.nii'))

    for run, fn in zip(runs, b0_fieldmaps_phase2):
        shutil.copy(fn, op.join(fmap_dir, f'sub-{subject:02d}_run-{run}_phase2.nii'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--subject', default=1, type=str)
    parser.add_argument('--bids_folder', default='/Users/hugofluhr/phd_local/data/LearningHabits/bids_formatted/ds-learninghabits')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)
