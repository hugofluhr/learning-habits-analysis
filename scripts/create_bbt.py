"""
Create a Big Behavior Table (bbt) from modeling data CSVs.

Configure the two variables below and run:
    python scripts/create_bbt.py

Run once per model (reduced / combined).
"""

import numpy as np
import pandas as pd
import sys
sys.path.append('/home/ubuntu/repos/learning-habits-analysis')
from utils.data import Subject, load_participant_list, STIM_REWARDS, STIM_FREQU

# ── Config ────────────────────────────────────────────────────────────────────
base_dir     = '/home/ubuntu/data/learning-habits'
modeling_dir = 'modeling_data/2024-09-27'   # change for combined
output_path  = '/home/ubuntu/data/learning-habits/bbt_062026_mf_cols.csv'  # change for combined
# ──────────────────────────────────────────────────────────────────────────────

sub_ids  = load_participant_list(base_dir)
subjects = [
    Subject(base_dir, sub_id, include_modeling=True, include_imaging=False, modeling_dir=modeling_dir)
    for sub_id in sub_ids
]

bbt = pd.concat(
    [
        pd.concat(
            [pd.DataFrame({'sub_id': [sub.sub_id] * len(sub.extended_trials)}),
             sub.extended_trials.reset_index(drop=True)],
            axis=1
        )
        for sub in subjects
    ],
    ignore_index=True
)

# Fix stim_chosen / stim_unchosen (3-trial mismatch in source behavioral data)
bbt_resp = bbt[~bbt['action'].isna()]
chosen_stim   = bbt_resp.left_stim.where(bbt_resp.action == 1, bbt_resp.right_stim).astype(float)
unchosen_stim = bbt_resp.right_stim.where(bbt_resp.action == 1, bbt_resp.left_stim).astype(float)
bbt.loc[bbt_resp.index, 'stim_chosen']   = chosen_stim
bbt.loc[bbt_resp.index, 'stim_unchosen'] = unchosen_stim
bbt_resp = bbt.loc[bbt['action'].notna()]

# Chosen / unchosen stimulus values
def _get_stim_value(row, stim_col, value_kind):
    stim = row[stim_col]
    if pd.isna(stim):
        return np.nan
    return row.get(f"stim{int(stim)}_value_{value_kind}", np.nan)

bbt['chosen_value_rl']   = bbt.apply(lambda r: _get_stim_value(r, 'stim_chosen',   'rl'), axis=1)
bbt['chosen_value_ck']   = bbt.apply(lambda r: _get_stim_value(r, 'stim_chosen',   'ck'), axis=1)
bbt['unchosen_value_rl'] = bbt.apply(lambda r: _get_stim_value(r, 'stim_unchosen', 'rl'), axis=1)
bbt['unchosen_value_ck'] = bbt.apply(lambda r: _get_stim_value(r, 'stim_unchosen', 'ck'), axis=1)

# Model-free chosen stimulus value and frequency
bbt['chosen_stim_value'] = bbt['stim_chosen'].apply(
    lambda x: STIM_REWARDS[int(x)] if not pd.isna(x) else np.nan
)
bbt['chosen_stim_frequ'] = bbt['stim_chosen'].apply(
    lambda x: STIM_FREQU[int(x)] if not pd.isna(x) else np.nan
)

# Combined choice value for chosen stimulus
beta_rl_col = [c for c in bbt.columns if c.startswith('beta_rl')][0]
beta_ck_col = [c for c in bbt.columns if c.startswith('beta_ck')][0]
bbt['chosen_choice_val'] = (
    bbt[beta_rl_col] * bbt['chosen_value_rl'] +
    bbt[beta_ck_col] * bbt['chosen_value_ck']
)

# Z-score within subject
columns_to_normalize = [
    'reward',
    'first_stim_value_rl',  'second_stim_value_rl',
    'first_stim_value_ck',  'second_stim_value_ck',
    'first_stim_choice_val', 'second_stim_choice_val',
    'chosen_value_rl',  'chosen_value_ck',
    'unchosen_value_rl', 'unchosen_value_ck',
    'chosen_choice_val',
    'chosen_stim_value',
]

for col in columns_to_normalize:
    bbt[col + '_zscore'] = (
        bbt.groupby('sub_id')[col]
           .transform(lambda x: (x - x.mean()) / x.std())
    )

# Verify z-scoring (skip subjects with zero variance on CK values)
zero_ck_subs = bbt.groupby('sub_id')['chosen_value_ck'].std()
zero_ck_subs = set(zero_ck_subs[zero_ck_subs == 0].index)
if zero_ck_subs:
    print(f"Skipping z-score check for subjects with zero CK variance: {zero_ck_subs}")

for sub_id, group in bbt.groupby('sub_id'):
    if sub_id in zero_ck_subs:
        continue
    for col in columns_to_normalize:
        col_z = col + '_zscore'
        mean = np.nanmean(group[col_z])
        std  = group[col_z].std()
        assert np.isclose(mean, 0, atol=1e-6), f"Mean for {col_z} in {sub_id} is not zero: {mean}"
        assert np.isclose(std,  1, atol=1e-6), f"Std for {col_z} in {sub_id} is not one: {std}"

bbt.to_csv(output_path, index=False)
print(f"Saved {len(bbt)} rows to {output_path}")
