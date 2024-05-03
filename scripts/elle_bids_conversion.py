import re
import os
import os.path as op
import shutil
import gzip
import argparse
import pandas as pd
import glob
from nilearn import image
import numpy as np
import json


def main(subject, bids_folder='/data'):
    #sourcedata_root = op.join(bids_folder, 'sourcedata', 'fmri',
    #                          f'SNS_MRI_MLEARN_S{subject:05d}_01')
    
    sourcedata_folder1 = '//idnas32.uzh.ch/g_econ_rawdata$/rawdata/2022'
    sourcedata_folder2 = '//idnas32.uzh.ch/g_econ_rawdata$/rawdata/2023'

    sourcedata_root1 = [os.path.join(root, dir) for root, dirs, files in os.walk(sourcedata_folder1) for dir in dirs if dir == f'SNS_MRI_MLEARN_S{subject:05d}_01']
    sourcedata_root2 = [os.path.join(root, dir) for root, dirs, files in os.walk(sourcedata_folder2) for dir in dirs if dir == f'SNS_MRI_MLEARN_S{subject:05d}_01']

    if sourcedata_root1:
        sourcedata_root = sourcedata_root1[0]
    elif sourcedata_root2:
        sourcedata_root = sourcedata_root2[0]
    else:
        print("folder not found")

    # *** ANATOMICAL DATA ***
    # So not vt1w, which are reconstructed at different angle
    t1w = glob.glob(op.join(sourcedata_root, '*_t1w*.nii'))
    print(op.join(sourcedata_root, '*_t1w*.nii'))
    assert (len(t1w) != 0), "No T1w {t1w}"

    # Not present
    flair = glob.glob(op.join(sourcedata_root, '*flair*.nii'))
    if len(flair) != 1:
        print(f"More than 1/no FLAIR {flair}!")

    target_dir = op.join(bids_folder, 'ds-mlearn', f'sub-{subject:02d}', 'anat')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    if len(t1w) == 1:
        shutil.copy(t1w[0], op.join(target_dir, f'sub-{subject:02d}_T1w.nii'))
    else:
        for run0, t in enumerate(t1w):
            print(t)
            shutil.copy(t1w[0], op.join(target_dir, f'sub-{subject:02d}_run-{run0 + 1}_T1w.nii'))

    if len(flair) == 1:
        shutil.copy(flair[0], op.join(target_dir, f'sub-{subject:02d}_FLAIR.nii'))


    # # *** FUNCTIONAL DATA ***
    with open(op.abspath('./bold_template.json'), 'r') as f:
        json_template = json.load(f)
        print(json_template)

    reg = re.compile('.*run(?P<run>[0-9]+).*')
    funcs = glob.glob(op.join(sourcedata_root, '*run*.nii'))
    runs = [int(reg.match(fn).group(1)) for fn in funcs]

    target_dir = op.join(bids_folder, 'ds-mlearn', f'sub-{subject:02d}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for run, fn in zip(runs, funcs):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_task-learn_run-{run}_bold.nii'))

        json_sidecar = json_template
        #json_sidecar['PhaseEncodingDirection'] = 'i' if (run % 2 == 1) else 'i-'

        with open(op.join(target_dir, f'sub-{subject:02d}_task-learn_run-{run}_bold.json'), 'w') as f:
            json.dump(json_sidecar, f)

    # *** physio logfiles ***
    physiologs = glob.glob(op.join(sourcedata_root, '*run*scanphyslog*.log'))
    runs = [int(reg.match(fn).group(1)) for fn in physiologs]

    for run, fn in zip(runs, physiologs):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_task-learn_run-{run}_physio.log'))

    # *** Fieldmaps ***
    target_dir = op.join(bids_folder, 'ds-mlearn', f'sub-{subject:02d}', 'fmap')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    b0_fieldmaps_mag1 = glob.glob(op.join(sourcedata_root, '*fieldmapclea_ec1*_typ0.nii'))
    b0_fieldmaps_mag2 = glob.glob(op.join(sourcedata_root, '*fieldmapclea_ec2*_typ0.nii'))
    b0_fieldmaps_phase1 = glob.glob(op.join(sourcedata_root, '*fieldmapclea_ec1*_typ3.nii'))
    b0_fieldmaps_phase2 = glob.glob(op.join(sourcedata_root, '*fieldmapclea_ec2*_typ3.nii'))

    physiologs = glob.glob(op.join(sourcedata_root, '*fieldmap*scanphyslog*.log'))


    runs = [i+1 for i in range(len(b0_fieldmaps_mag1))]

    for run, fn in zip(runs, physiologs):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_run-{run}_physio.log'))

    for run, fn in zip(runs, b0_fieldmaps_mag1):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_run-{run}_magnitude1.nii'))
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


        with open(op.join(target_dir, f'sub-{subject:02d}_run-{run}_magnitude1.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.0046
            json.dump(json_sidecar, f)

        with open(op.join(target_dir, f'sub-{subject:02d}_run-{run}_magnitude2.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.006940000000000001
            json.dump(json_sidecar, f)

        with open(op.join(target_dir, f'sub-{subject:02d}_run-{run}_phase1.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.0046
            json.dump(json_sidecar, f)

        with open(op.join(target_dir, f'sub-{subject:02d}_run-{run}_phase2.json'), 'w') as f:
            json_sidecar['EchoTime'] = 0.006940000000000001
            json.dump(json_sidecar, f)

    for run, fn in zip(runs, b0_fieldmaps_mag2):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_run-{run}_magnitude2.nii'))

    for run, fn in zip(runs, b0_fieldmaps_phase1):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_run-{run}_phase1.nii'))

    for run, fn in zip(runs, b0_fieldmaps_phase2):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_run-{run}_phase2.nii'))

    # *** DTI ***
    target_dir = op.join(bids_folder, 'ds-mlearn', f'sub-{subject:02d}', 'dwi')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    dwi1 = glob.glob(op.join(sourcedata_root, '*bva_bvalue1*.nii'))
    dwi2 = glob.glob(op.join(sourcedata_root, '*bva_bvalue2*.nii'))

    for fn in dwi1:
        nr = re.split(r'(\d+)', str(fn))[-2]
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_dir-1_{nr}_dwi.nii'))

    for fn in dwi2:
        nr = re.split(r'(\d+)', str(fn))[-2]
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_dir-2_{nr}_dwi.nii'))

    dwi_log = glob.glob(op.join(sourcedata_root, '*bva.log'))
    shutil.copy(dwi_log[0], op.join(target_dir,f'sub-{subject:02d}_dwi.log'))
    dwi_par = glob.glob(op.join(sourcedata_root, '*bva.par'))
    shutil.copy(dwi_par[0], op.join(target_dir, f'sub-{subject:02d}dir-_dwi.json'))
    #physio_json = glob.glob(op.join(sourcedata_root, '*bva*scanphyslog*.json'))
    #shutil.copy(physio_json[0], f'sub-{subject:02d}dir-_physio.json')
    #physio_log = glob.glob(op.join(sourcedata_root, '*bva*scanphyslog*.log'))
    #shutil.copy(physio_log[0], f'sub-{subject:02d}dir-_physio.log')
    #physio_tsv = glob.glob(op.join(sourcedata_root, '*bva*scanphyslog*.tsv'))
    #shutil.copy(physio_tsv[0], f'sub-{subject:02d}dir-_physio.tsv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('--bids_folder', default='D:/data')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)