import os
import sys
import time
import json
import pickle
import warnings
from datetime import datetime


from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import pandas as pd
from nilearn.image import load_img, concat_imgs
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.reporting import make_glm_report

sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import sort_key
from utils.analysis import compute_parametric_modulator, orthogonalize_modulator

# Dynamically set the number of workers based on available CPUs
MAX_WORKERS = 6

BASE_DIR = '/home/ubuntu/data/social-risk'
SUB_IDS = ['01', '02', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
           '14', '16', '17', '18', '19', '22', '23', '25']

model_params = {
    'model_name': 'glm2',
    'tr': 3.,
    'hrf_model': 'spm + derivative',
    'drift_model': 'cosine',
    'high_pass': 1./128,
    'noise_model': 'ar1',
    'smoothing_fwhm': None,
    'duration': 'all',
}

def create_events(behav_offset):
    # Info event
    df_info = pd.DataFrame({
        'trial_type': 'Info',
        'onset': behav_offset['onsetINF'],
        'duration': 0,
        'run': behav_offset.index.get_level_values(0),
        'trial': behav_offset.index.get_level_values(1)
    })
    # Opt1 event
    df_opt1 = pd.DataFrame({
        'trial_type': 'Opt1',
        'onset': behav_offset['onsetOPT1'],
        'duration': 0,
        'run': behav_offset.index.get_level_values(0),
        'trial': behav_offset.index.get_level_values(1),
        'Opt1xWTP': behav_offset.loc[:, 'WTPOpt1']
    })

    # Opt2 event
    df_opt2 = pd.DataFrame({
        'trial_type': 'Opt2',
        'onset': behav_offset['onsetOPT2'],
        'duration': 0,
        'run': behav_offset.index.get_level_values(0),
        'trial': behav_offset.index.get_level_values(1),
        'Opt2xWTP': behav_offset.loc[:, 'WTPOpt2']
    })

    # Decision event
    df_decision = pd.DataFrame({
        'trial_type': 'Decision',
        'onset': behav_offset['onsetDEC'],
        'duration': 0,
        'run': behav_offset.index.get_level_values(0),
        'trial': behav_offset.index.get_level_values(1)
    })

    # Feedback event
    df_feedback = pd.DataFrame({
        'trial_type': 'Feedback',
        'onset': behav_offset['onsetFBK'],
        'duration': 0,
        'run': behav_offset.index.get_level_values(0),
        'trial': behav_offset.index.get_level_values(1)
    })

    # Put it all together
    events = pd.concat([df_info, df_opt1, df_opt2, df_decision, df_feedback],
                    ignore_index=True)

    events = events.sort_values(by='onset').reset_index(drop=True)
    # split sessions
    events['trial_type'] = 'Sn_' + events['run'].astype(str) + '_' + events['trial_type']
    return events

def create_drift_matrix(frametimes, drift_model, high_pass):
    drift_Xs = [make_first_level_design_matrix(frametimes[s],
                                   drift_model=drift_model,
                                   high_pass=high_pass)
                                   for s in range(3)]
    drift_Xs = [drift_Xs.drop(columns=['constant']) for drift_Xs in drift_Xs]
    drift_Xs = [df.add_prefix(f'Sn_{i+1}_') for i, df in enumerate(drift_Xs)]
    drift_X = drift_Xs[0].join([drift_Xs[1], drift_Xs[2]], how='outer').fillna(0.)
    return drift_X


def glm2(sub_id, behav_df, model_params):

    # Parameters
    model_name = model_params['model_name']
    tr = model_params['tr']
    hrf_model = model_params['hrf_model']
    noise_model = model_params['noise_model']
    smoothing_fwhm = model_params['smoothing_fwhm']
    drift_model = model_params['drift_model']
    high_pass = model_params['high_pass']
    duration = model_params['duration']

    # Subject directory
    sub_dir = os.path.join(BASE_DIR, sub_id, 'func')

    # Create output directory
    derivatives_dir = os.path.join(BASE_DIR, 'nilearn')
    current_time = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(derivatives_dir, f"{model_name}_{current_time}")
    sub_output_dir = os.path.join(model_dir, sub_id)
    if not os.path.exists(sub_output_dir):
        os.makedirs(sub_output_dir)

    # Load fMRI volume
    runs = [os.path.join(sub_dir, f'swra{sub_id}_sess{i}.nii') for i in range(1,4)]
    imgs = [load_img(run) for run in runs]
    n_scans = [img.shape[-1] for img in imgs]

    # Frametimes and offsets (for concatenation)
    ft1 = np.arange(n_scans[0]) * tr
    ft2 = np.arange(n_scans[1]) * tr + n_scans[0] * tr
    ft3 = np.arange(n_scans[2]) * tr + (n_scans[0] + n_scans[1]) * tr
    frametimes = np.concatenate([ft1, ft2, ft3])
    offsets = [0, n_scans[0] * tr, (n_scans[0] + n_scans[1]) * tr]

    # Load confounds
    confounds = [os.path.join(sub_dir, f'rp_a{sub_id}_sess{i}.txt') for i in range(1,4)]
    confounds = [
        pd.read_csv(
            confound, 
            sep=r'\s+', 
            names=[f'Sn_{i+1}_R{j}' for j in range(1, 7)]
        ) 
        for i, confound in enumerate(confounds)
    ]
    # demean confounds and put them together
    for i in range(3):
        confounds[i] = confounds[i] - confounds[i].mean()
    all_confounds = pd.concat(confounds, ignore_index=True).fillna(0.)

    # No brain mask here
    brain_mask = None

    # Load events and apply offsets
    behav_offset = behav_df.copy()
    for i, run in enumerate([1, 2, 3]):
        mask = behav_offset.index.get_level_values(0) == run
        for col in behav_offset.columns[behav_offset.columns.str.contains('onset')]:
            behav_offset.loc[mask, col] = behav_offset.loc[mask, col] + offsets[i]
    events = create_events(behav_offset)

    # Ignore warnings related to null duration events and unexpected columns in 
    warnings.filterwarnings("ignore", message=".*events with null duration.*")
    warnings.filterwarnings("ignore", 
                            message=".*following unexpected columns in events data.*")
    
    # Create design matrix
    X = make_first_level_design_matrix(frametimes,
                                    events, 
                                    hrf_model=hrf_model, 
                                    drift_model=None,
                                    add_regs=all_confounds)
    X = X.drop('constant', axis=1)
    # spm and nilearn have different naming conventions
    # Rename columns to match SPM
    X.columns = X.columns.str.replace('_derivative', 'dt')

    # Drift matrix
    drift_X = create_drift_matrix([ft1, ft2, ft3], drift_model, high_pass)
    
    # Parametric modulation
    parametric_modulators = {}
    for s in range(1, 4):
        conditions = [
            (f'Sn_{s}_Opt1', 'Opt1xWTP'),
            (f'Sn_{s}_Opt2', 'Opt2xWTP')
        ]

        for condition, modulator in conditions:
            mod_name = f'Sn_{s}_'+modulator
            pm = compute_parametric_modulator(events, condition, modulator, frametimes, 
                                              hrf_model, normalize='center')
            pm_ortho = orthogonalize_modulator(pm[:, 0], X[condition])
            parametric_modulators[mod_name] = pd.DataFrame({mod_name: pm_ortho,
                                                            mod_name+'dt': pm[:, 1]})

    all_modulators = pd.concat(parametric_modulators.values(), axis=1)
    all_modulators.index = X.index

    X = pd.concat([X, all_modulators], axis=1)
    # Sorting columns
    X = X[sorted(X.columns, key=lambda col: (col.split('_')[1], col))]
    # Sort same as spm design matrix
    new_order = sorted(X.columns[:20], key=sort_key)
    new_order = (
        new_order + [col.replace('Sn_1', 'Sn_2') for col in new_order] 
        + [col.replace('Sn_1', 'Sn_3') for col in new_order]
    )
    X = X[new_order]
    # Add session dummy variables and drifts to the design matrix
    session_dummy = np.eye(3).repeat(n_scans, axis=0)
    session_df = pd.DataFrame(session_dummy, 
                              columns=[f'Sn_{i}_constant' for i in range(1, 4)],
                                index=X.index)
    X = pd.concat([X, drift_X, session_df], axis=1)

    # Create the model
    model = FirstLevelModel(t_r=tr, 
                            smoothing_fwhm=None, 
                            mask_img=brain_mask,
                            hrf_model=hrf_model,
                            noise_model=noise_model,
                            drift_model=None, # handled in design matrix
                            minimize_memory=False) 
    
    # concatenate images
    concatenated_img = concat_imgs(imgs)

    model = model.fit(concatenated_img, design_matrices=X, sample_masks=None)

    # Save the events dataframe to csv
    events_path = os.path.join(sub_output_dir, f'{sub_id}_events.csv')
    events.to_csv(events_path, index=False)
    #print(f"Events saved to {events_path}")

    # Save the fitted model using pickle, not doing this because it is too large
    # model_filename = os.path.join(sub_output_dir, f'{sub_id}_model.pkl')
    # with open(model_filename, 'wb') as f:
    #     pickle.dump(model, f)
    #print(f"GLM model saved to {model_filename}")

    # Save beta maps for each regressor
    for i, column in enumerate(X.columns):
        beta_map = model.compute_contrast(np.eye(len(X.columns))[i], output_type='effect_size')
        beta_path = os.path.join(sub_output_dir, f'{sub_id}_betamap_{column}.nii.gz')
        beta_map.to_filename(beta_path)
        #print(f"Saved: {beta_path}")

    # Save the R2 map
    r2_map = model.r_square[0]
    r2_map_path = os.path.join(sub_output_dir, f'{sub_id}_r2map.nii.gz')
    r2_map.to_filename(r2_map_path)

    # Contrasts
    connames = ['Info','Infodt','Opt1','Opt1dt','Opt1xWTP','Opt1xWTPdt',
                'Opt2','Opt2dt','Opt2xWTP','Opt2xWTPdt',
                'Decision','Decisiondt','Feedback','Feedbackdt']
    
    contrasts = {}
    for name in connames:
        weights = np.array([1 if col.endswith(name) else 0 for col in X.columns])
        n_matches = np.sum(weights)
        if n_matches > 0:
            contrasts[name] = weights / n_matches
        else:
            print(f"Warning: No columns found for contrast {name}")

    for con in contrasts:
        z_map = model.compute_contrast(contrasts[con], output_type='z_score')
        beta_map = model.compute_contrast(contrasts[con], output_type='effect_size')
        stat_map = model.compute_contrast(contrasts[con], output_type='stat')
        # save the contrast as zmap and betamap
        z_map.to_filename(os.path.join(sub_output_dir, f'con_{con}_zmap.nii.gz'))
        beta_map.to_filename(os.path.join(sub_output_dir, f'con_{con}_betamap.nii.gz'))
        stat_map.to_filename(os.path.join(sub_output_dir, f'con_{con}_statmap.nii.gz'))

    # Save the design matrix
    design_matrix_path = os.path.join(sub_output_dir, f'{sub_id}_design_matrix.csv')
    X.to_csv(design_matrix_path, index=False)
    #print(f"Design matrix saved to {design_matrix_path}")

    # Save analysis parameters
    params_path = os.path.join(sub_output_dir, f'{sub_id}_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_params, f, indent=4)
    #print(f"Analysis parameters saved to {params_path}")


def process_subject(sub_id, behav_df, model_params):
    print(f"Processing Subject {sub_id}...")  
    try:
        glm2(sub_id, behav_df, model_params)
        return f"Subject {sub_id} processed successfully"
    except Exception as e:
        return f"An error occurred for Subject {sub_id}: {e}"

if __name__ == "__main__":

    # Load data common to all subjects
    subs_ids = ['SUBJ_' + sub_id for sub_id in SUB_IDS]
    behav_path = os.path.join(BASE_DIR, 'analysis')
    bbt = pd.read_csv(os.path.join(behav_path, 'BevBigTable.csv'), index_col=[0,1,2])
    bbt.index = bbt.index.set_levels([bbt.index.levels[0].map(lambda x: f'SUBJ_{x:02d}')]
                                  + bbt.index.levels[1:])
    
    start_time = time.time()

    # Parallel processing with joblib
    results = Parallel(n_jobs=MAX_WORKERS)(
        delayed(process_subject)(sub, bbt.loc[sub], model_params) for sub in subs_ids
    )

    # Print results for each subject
    for result in results:
        print(result)

    end_time = time.time()
    print(f"Total time elapsed: {end_time - start_time:.2f} seconds")