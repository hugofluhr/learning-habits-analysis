from glmsingle.glmsingle import GLM_single
import argparse
import os
import os.path as op
from nilearn import image
from numrisk.utils.data import Subject
from nilearn.glm.first_level import make_first_level_design_matrix
import numpy as np
import pandas as pd
import nibabel as nib

import warnings
warnings.filterwarnings('ignore')

# run in terminal (weird numba error, which does not happen in VS-notebook):
# rm -rf ~/.numba_cache
# export NUMBA_DISABLE_JIT=1


# check: https://github.com/cvnlab/GLMsingle/blob/main/glmsingle/glmsingle.py

TR = 2.827 # that was wrong, and then design_matrix was weirdly of (but hard to figure out, random early volumes had multiple ones)
stim_duration = 0.6 

# open questions: 
# "....specifying that a given condition occurs more than one time over the course of the experiment, this information can and will be used for cross-validation purposes."
# name "trial_type" rather "n1_number" and "n2_number" - so some events are the same?

def get_fmri_events_bothStim(sub, session, runs, bids_folder):
    behavior = []
    for run in runs:
        behavior.append(pd.read_table(op.join(
            bids_folder, f'sub-{sub}/ses-{session}/func/sub-{sub}_ses-{session}_task-magjudge_run-{run}_events.tsv')))

    behavior = pd.concat(behavior, keys=runs, names=['run'])
    behavior = behavior.reset_index().set_index(
        ['run', 'trial_type'])

    behavior = behavior[behavior['trial_nr'] != 0]

    stimulus1 = behavior.xs('stimulus 1', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type', 'n1']]
    stimulus1['duration'] = stim_duration
    stimulus1['trial_type'] = stimulus1.trial_nr.map(lambda trial: f'trial_{trial:03d}_n1')

    stimulus2 = behavior.xs('stimulus 2', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_nr', 'trial_type', 'n2']]
    stimulus2['duration'] = stim_duration
    stimulus2['trial_type'] = stimulus2.trial_nr.map(lambda trial: f'trial_{trial:03d}_n2')

    events = pd.concat((stimulus1, stimulus2)).sort_index()
    events = events[['onset', 'duration', 'trial_type']]  
    
    return events 

# not there yet, also saving output step has to be adapted
def load_fmri_data(subject,bids_folder, space,session=1, task = 'magjudge', runs=range(1, 7)):
    """Load fMRI data from BIDS derivatives (supports NIfTI and GIFTI surface data)."""
    import nibabel as nib
    base = op.join(bids_folder, 'derivatives', 'fmriprep',f'sub-{subject}', f'ses-{session}', 'func')

    data = []
    for run in runs:
        if "fsaverage" in space:  # Surface data
            hemi_data = []
            for hemi in ['L', 'R']:
                path = op.join(base, f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_hemi-{hemi}_bold.func.gii')
                gii = nib.load(path)
                hemi_data.append(np.column_stack([d.data for d in gii.darrays]))
            data.append(np.vstack(hemi_data))  # Combine hemispheres
        else:  # Volume data
            path = op.join(base, f'sub-{subject}_ses-{session}_task-{task}_run-{run}_space-{space}_desc-preproc_bold.nii.gz')
            data.append(nib.load(path).get_fdata())
    return data 
##


def main(subject,  bids_folder, space,  runs = range(1, 7), session = 1, task='magjudge'): #, smoothed=False,  retroicor=False, split_data = None): # 'both', 'run_123', 'run_456'
    
    derivatives = op.join(bids_folder, 'derivatives')
    subject = f'{int(subject):02d}'

    key = f'glm_stim.denoise'
    base_dir = op.join(derivatives, key, f'sub-{subject}', f'ses-{session}', 'func')
    os.makedirs(base_dir, exist_ok=True)

    # get fMRI data
    im_data = load_fmri_data(subject, bids_folder=bids_folder, space=space) # _bold missing for numrisk

    # construct design matrix
    onsets = get_fmri_events_bothStim(subject, session, runs, bids_folder)
    tr = TR
    N_volumes = np.shape(im_data)[-1] # number of volumes
    frametimes = np.linspace(tr/2., (N_volumes - .5)*tr, N_volumes)
    onsets['onset'] = ((onsets['onset']+tr/2.) // tr) * tr
    dm = [make_first_level_design_matrix(frametimes, onsets.loc[run], hrf_model='fir', oversampling=100.,
                                         drift_order=0,
                                         drift_model=None).drop('constant', axis=1) for run in runs]
    dm = pd.concat(dm, keys=runs, names=['run']).fillna(0) # keys = range(1, 7)
    dm.columns = [c.replace('_delay_0', '') for c in dm.columns]
    dm /= dm.max()
    dm[dm < 1.0] = 0.0
    X = [dm.loc[run].values for run in runs]

    print("Design matrix and data shapes:")
    print(np.shape(X))
    print(np.shape(im_data))

    # set options for GLM-single
    opt = dict()
    opt['wantlibrary'] = 1 # set important fields for completeness (but these would be enabled by default)
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1
    opt['wantfileoutputs'] = [0, 0, 0, 1] # keep the relevant outputs in memory and also save them to the disk

    glmsingle_obj = GLM_single(opt)
    results_glmsingle = glmsingle_obj.fit(
        X,
        im_data,
        stim_duration,
        tr,
        outputdir=base_dir,
        figuredir = op.join(base_dir, 'GLMestimatesingletrialfigures') # would be written to cwd otherwise and could crash when multiple nodes use it a the same time 
        )
    
    # Save results: separate n1 and n2 betas
    betas = results_glmsingle['typed']['betasmd']
    if space == 'T1w':
        base = op.join(bids_folder, 'derivatives', 'fmriprep',f'sub-{subject}', f'ses-{session}', 'func')
        example_image = op.join(base, f'sub-{subject}_ses-{session}_task-{task}_run-1_space-{space}_desc-preproc_bold.nii.gz')
        betas = image.new_img_like(example_image, betas)

        # n1s
        betas_n1 = image.index_img(betas, slice(0, None, 2) ) # slice(0, None, 2) where to start, where to end, step size
        betas_n1.to_filename(op.join(base_dir, f'sub-{subject}_ses-{session}_task-magjudge_space-{space}_desc-stims1_pe.nii.gz'))
        # n2s
        betas_n2 = image.index_img(betas, slice(1, None, 2) ) # slice(0, None, 2) where to start, where to end, step size
        betas_n2.to_filename(op.join(base_dir, f'sub-{subject}_ses-{session}_task-magjudge_space-{space}_desc-stims2_pe.nii.gz'))
    elif space == 'fsaverage5':
        n_vertices_total, n_betas = betas.shape[:2]
        n_vertices_hemi = n_vertices_total // 2

        hemi_data = {
            'L': betas[:n_vertices_hemi],
            'R': betas[n_vertices_hemi:]
        }

        for hemi, hemi_betas in hemi_data.items():
            # Split N1 / N2
            hemi_betas_n1 = hemi_betas[..., ::2]
            hemi_betas_n2 = hemi_betas[..., 1::2]

            for stim_n, hemi_b in zip([1, 2], [hemi_betas_n1, hemi_betas_n2]):
                darrays = [nib.gifti.GiftiDataArray(hemi_b[:, i].astype(np.float32)) for i in range(hemi_b.shape[1])]
                gii = nib.GiftiImage(darrays=darrays)
                fn = op.join(base_dir, f'sub-{subject}_ses-{session}_task-magjudge_space-{space}_desc-stims{stim_n}_pe_hemi-{hemi}.gii')
                gii.to_filename(fn)
                print(f"Saved {fn}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/mnt_AdaBD_largefiles/Data/SMILE_DATA/DNumRisk/ds-numrisk')
    parser.add_argument('--space', default='fsaverage5') # 'T1w'


    args = parser.parse_args()
    main(args.subject,bids_folder=args.bids_folder, space=args.space)