from glmsingle.glmsingle import GLM_single
import argparse
import os
import os.path as op
from nilearn import image

from numloss.utils.data import Subject # project specific 

from nilearn.glm.first_level import make_first_level_design_matrix

import nibabel as nib
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from pathlib import Path

warnings.filterwarnings('ignore')

def extract_ses_run(fname):
    parts = fname.split('_')
    ses_val = None
    run_val = None
    for p in parts:
        if p.startswith('ses-'):
            ses_val = int(p.replace('ses-', '').replace('.nii.gz', ''))
        if p.startswith('run-'):
            run_val = int(p.replace('run-', '').replace('.nii.gz', ''))
    return ses_val, run_val

def main(subject, bids_folder, tr=None, n_volumes=None, n_runs=None, task='numloss', confounds=False, smoothed=False):

    # bids derivatives
    bids_folder = Path(bids_folder)
    derivatives = bids_folder / 'derivatives'

    # subject handeling 
    sub = Subject(subject, bids_folder=bids_folder)     
    runs = sub.get_runs(None, n_runs=n_runs, task=task) 
    base_dir = 'glm_stim1.denoise'
    if smoothed:
        base_dir += '.smoothed'

    # specify onsets
    onsets = sub.get_onsets(None, n_runs=n_runs, task=task)
    if onsets.index.nlevels == 4 and onsets.index.names[:2] == ['session','session']:
        print(">>> FIXING onsets index (duplicate session level + string sessions)")
        onsets = onsets.copy()
        onsets.index = onsets.index.droplevel(1)  # remove extra session
        onsets.index = onsets.index.set_levels([
            onsets.index.levels[0].astype(int),   # session
            onsets.index.levels[1].astype(int),   # run
            onsets.index.levels[2]                # trial_nr
        ])

    # design matrix must have single_trial onsets x condition (not single trial) labels 
    # b/c toolbox uses this info within cros validation 
    stim1_mask = onsets['trial_type'] == 'stimulus1' # label stimulus conditions by n1/n2 magnitudes
    onsets.loc[stim1_mask, 'trial_type'] = 'stimulus1_' + onsets.loc[stim1_mask, 'n1'].astype(str)
    stim2_mask = onsets['trial_type'] == 'stimulus2'
    onsets.loc[stim2_mask, 'trial_type'] = 'stimulus2_' + onsets.loc[stim2_mask, 'n2'].astype(str)
    unique_trial_types = onsets.trial_type.unique()
    tt2idx = {tt: idx for idx, tt in enumerate(unique_trial_types)}

    # instanteause events # TODO 
    onsets['duration'] = 0.0

    # ? TODO
    img0 = None
    Xs = []
    datas = []
    sess_ind = []
    colnames = None

    # preprocessing files
    bold_files = sub.get_preprocessed_bold(session=None, n_runs=n_runs, task=task)

    # 
    for bold_path in tqdm(bold_files):
        fname = op.basename(bold_path)
        
        # load nifty img
        img = image.load_img(bold_path)
        if img0 is None:
            img0 = img  # keep header for saving later
            if tr is None:
                tr = img0.header.get_zooms()[-1]
                
        if smoothed:
            img = image.smooth_img(img, fwhm=5.0)

        # typically: 01_01 etc.
        ses_val, run_val = extract_ses_run(fname) # TODO check func
        if ses_val is None or run_val is None:
            raise ValueError(f"File does not contain ses/run: {fname}")
        key = (ses_val, run_val)        

        # saftey check 
        if (ses_val, run_val) not in onsets.index.droplevel('trial_nr').unique():
            print(f"[INFO] Skipping run ses-{ses_val} run-{run_val}: no events")
            continue

        # event dataframe (onset x task parameters)
        evt = onsets.xs((ses_val, run_val), level=('session','run'))
        if evt.empty:
            print(f"[INFO] Skipping run ses-{ses_val} run-{run_val}: events are empty")
            continue
        evt = evt.copy()
        evt.reset_index(inplace=True)
        # evt['onset'] = np.round(evt['onset'] / tr) * tr + tr/2.0

        # nilearn: read img data
        data_i = img.get_fdata()  # func data: 4D x-y-z-time (time: volumes)
        nvol_i = data_i.shape[-1]

        # infer frametimes # TODO understand frametimes
        frametimes = np.linspace(tr/2., (nvol_i - .5)*tr, nvol_i) 

        # create design matrix (volumes x condition) -> final dm [0,1]
        dm_i = np.zeros((nvol_i, len(unique_trial_types)))
        for i, row in evt.iterrows():
            onset = row['onset']
            trial_type = row['trial_type']        
            onset_idx = int(np.round(onset / tr)) # TODO 
            trial_type_idx = tt2idx[trial_type]
            dm_i[onset_idx, trial_type_idx] = 1.0 

        
        Xs.append(dm_i)
        datas.append(data_i)
        sess_ind.append(ses_val)

    X = Xs       # lst of design mats (per run/session)
    data = datas # nifty image data
    print(f"Collected {len(X)} runs; example DM shape: {X[0].shape}, example data shape: {data[0].shape}")
    for x in X:
        print(x.shape)

    # out location
    base_dir = op.join(derivatives, base_dir, f'sub-{subject}', 'func')
    os.makedirs(base_dir, exist_ok=True)


    #--- specify GLMsingle ---
    opt = dict()
    opt['sessionindicator'] = np.array(sess_ind, dtype=int)[np.newaxis, :]
    #opt['sessionindicator'] = np.array([session for (session, run), d in dm.groupby(['session', 'run'])])[np.newaxis, :]
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1
    opt['wantfileoutputs'] = [0, 0, 0, 1]
    stimdur_scalar = 0.6 # duration of array1 and array2 presentation
    glmsingle_obj = GLM_single(opt)

    # out loc
    figuredir = Path(bids_folder) / Path(base_dir) / 'figures'
    figuredir.mkdir(parents=True, exist_ok=True)

    #--- fit model ---
    results_glmsingle = glmsingle_obj.fit(
        X,
        data,
        stimdur_scalar,
        tr,
        outputdir=base_dir,
        figuredir=str(figuredir))

    #extract and save betas for stimulus1 and stimulus2 conditions
    betas = results_glmsingle['typed']['betasmd']
    betas = image.new_img_like(img0, betas)

    stim1_betas = image.index_img(betas, slice(None, None, 2))
    stim2_betas = image.index_img(betas, slice(1, None, 2))

    fn_template = op.join(base_dir, 'sub-{subject}_task-{task}_space-T1w_desc-{par}_pe.nii.gz')
    
    stim1_betas.to_filename(fn_template.format(subject=subject, task=task, par='stim1'))
    stim2_betas.to_filename(fn_template.format(subject=subject, task=task, par='stim2'))

    r2 = results_glmsingle['typed']['R2']
    r2 = image.new_img_like(img0, r2)
    r2.to_filename(fn_template.format(subject=subject, task=task, par='R2'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-numloss')
    parser.add_argument('--tr', default=None, type=float)
    parser.add_argument('--n_volumes', default=None, type=int)
    parser.add_argument('--n_runs', default=10, type=int)
    parser.add_argument('--task', default='numloss', type=str)
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder, tr=args.tr, n_volumes=args.n_volumes, n_runs=args.n_runs, task=args.task, smoothed=args.smoothed)
