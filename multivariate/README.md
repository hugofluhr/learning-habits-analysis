# Multivariate (MVPA) pipeline

Single-trial fMRI decoding for the reward-learning habits study. One GLMsingle step
estimates per-trial beta maps; every downstream analysis (whole-brain/ROI decoding,
searchlight, FREM, RSA) consumes the same betas.

All beta maps are locked to **first-stimulus (cue) onset** — one volume per presentation,
8 stimulus identities, emitted in chronological order (`learning1` → `learning2` → `test`,
each sorted by `t_first_stim`). Sessions: `learning1`, `learning2`, `test`.

## Pipeline overview

```
run_glmsingle.py ──► sub-<id>_glmSingle_betas_CUES.nii.gz   (x,y,z,n_trials)
                     sub-<id>_glmSingle_betas_CUES_info.csv  (trial → stim_cat / run)
                          │
      ┌───────────────────┼───────────────────────────────┐
      ▼                   ▼                                ▼
  category            category                        beta-version QC
  decoding /          FREM                             (B/C/D, wholebrain
  searchlight                                           + visual-cortex ROI)
```

**Stimulus category** (`stim_cat`, 4-way, chance 0.25) — a strong, well-decoded signal.
Used as the readout of interest *and* as the validation probe for the beta-version QC.
Label comes straight from the betas' info CSV.

> Reward/value decoding (objective reward level of the cue, sourced from the BBT) is
> under development on the `qvalue-decoding` branch — not yet merged, so not documented
> here. See that branch's `run_qvalue_searchlight.py` / `run_qvalue_frem.py` once merged.

## Environment

- **Local** (per repo `CLAUDE.md`): the `neuroim` conda env — `conda run -n neuroim python ...`.
- **Cluster** (UZH sciencecluster): env at `~/data/conda/envs/learning-habits`, built with
  `bash multivariate/build_env.sh` (SLURM job; installs `environment.yml` + `glmsingle`).

## How to run

Subjects always come from `participants_mvpa.tsv` (the canonical MVPA sample) — never a
`seq` range. Pass explicit IDs only when you deliberately want a subset. Every `run_*.py`
is single-subject and **skips subjects already done** (pass `--overwrite` to force), so the
drivers are safe to resume.

### On the cluster (SLURM, preferred)

```bash
bash multivariate/submit_glmsingle.sh                 # 1. estimate betas (run first)
bash multivariate/submit_searchlight.sh               # 2. category searchlight
bash multivariate/submit_decoding.sh                  #    category WB + visual-ROI decoding
bash multivariate/submit_beta_qc_decoding.sh          # 3. beta-version QC (category probe)
bash multivariate/submit_label_shuffle_qc.sh          # 4. label-shuffle robustness check

bash multivariate/submit_searchlight.sh 01 05 12      # specific subjects
```

`submit_glmsingle.sh`/`submit_searchlight.sh`/`submit_decoding.sh`/`submit_label_shuffle_qc.sh`
are SLURM **array** jobs (one task per subject; `submit_label_shuffle_qc.sh` throttles
concurrency with `%THROTTLE`, default 20, as a courtesy to other partition users).
`submit_beta_qc_decoding.sh` is a **single** job that loops all subjects internally via
`xargs -P` — measured per-subject cost is light (~35-45s), so an array job's per-task
scheduling overhead isn't worth it there; see the script's header comment for the
measurement and reasoning (label-shuffle QC used to follow this pattern too, until its
much heavier per-subject cost — ~14min — made genuine array concurrency the better trade,
see `submit_label_shuffle_qc.sh`'s header). `OVERWRITE=1 bash multivariate/submit_beta_qc_decoding.sh`
forces a rerun of subjects that already have output (e.g. after a script/schema change); same
for `submit_label_shuffle_qc.sh`.

### Locally / on the VM (xargs -P driver) — fallback when no scheduler

```bash
bash multivariate/run_local.sh <pipeline> [SUBJECT ...]
# pipelines: glmsingle | searchlight | decoding | frem | beta_qc
NPROC=8 THREADS=3 bash multivariate/run_local.sh glmsingle   # override concurrency
```

`run_local.sh` encodes per-pipeline concurrency (glmsingle is memory-bound → few subjects
at a time, several threads each; searchlight/decoding/frem/beta_qc are light/CPU-bound →
many subjects at once, single-threaded). Edit the `*_DIR` / `BASE_DIR` / `PY` variables at
the top to match the host.

### Stage → script map

| Stage | Runner | Cluster submit | `run_local` | Output |
|-------|--------|----------------|-------------|--------|
| Single-trial betas | `run_glmsingle.py` | `submit_glmsingle.sh` | `glmsingle` | `derivatives/glmsingle/` |
| Category searchlight | `run_searchlight.py` | `submit_searchlight.sh` | `searchlight` | `..._searchlight_stim_cat*.nii.gz` |
| Category WB + visual ROI | `run_decoding.py` | `submit_decoding.sh` | `decoding` | `..._decoding_accuracy.csv`, confusions |
| Category FREM | `run_frem.py` | *(run_local only)* | `frem` | `..._frem_coef_<cat>.nii.gz`, AUC |
| **Beta-version QC** | `run_beta_qc_decoding.py` | `submit_beta_qc_decoding.sh` (single job) | `beta_qc` | `..._beta_qc_decoding.csv` |
| **Label-shuffle QC** | `run_label_shuffle_qc.py` | `submit_label_shuffle_qc.sh` (array job) | *(cluster only)* | `..._label_shuffle_qc.csv` |

### Prerequisites / knobs

- **Category decoding and beta-version QC** both need a visual-cortex ROI. Build it once
  and place it in the decoding output dir:
  `python multivariate/build_visual_cortex_mask.py --output-dir <path>`
  (Harvard-Oxford atlas, MNI152 2mm — occipital + fusiform/temporal-occipital regions, i.e.
  early visual cortex through ventral temporal cortex, not just V1; that's deliberate,
  since category selectivity for real-world objects/faces lives predominantly in ventral
  temporal cortex).
- **Beta-version QC** decodes `stim_cat` (the validation probe) across GLMsingle model types
  B (+FITHRF) → C (+GLMdenoise) → D (+ridge). **Type A (ONOFF) is written by GLMsingle but
  skipped here** — it pools every event into a single on/off beta per voxel, so there's no
  per-trial signal to decode. Runs on **both** a whole-brain mask and the visual-cortex ROI
  (`--visual-cortex-mask`, required) — the ROI is the more diagnostic of the two, since
  whole-brain dilutes the category signal with tens of thousands of non-visual voxels, which
  can swamp subtler beta-version differences given only 328 trials. Reads `TYPE{B,C,D}.npy`
  directly and does not touch the production GLMsingle pipeline; `standardize=True`
  neutralizes type-D ridge shrinkage. B→D is usually but not guaranteed monotonic.
- **Label-shuffle QC** is a permutation-test negative control on the *production* decoding
  (type-D betas, same masks/CV as `run_decoding.py`): decodes `stim_cat` once with the true
  labels, then `--n-permutations` times (default 100) with labels shuffled globally across
  all trials before the same `LeaveOneGroupOut` CV. If real accuracy sits outside the
  shuffled-accuracy null distribution, that supports the category signal being genuine; if
  shuffled accuracy doesn't collapse to chance, that's a leakage red flag (e.g.
  standardization fit before the CV split, autocorrelation between adjacent trials). Unlike
  `run_decoding.py`, uses `standardize=True` on both masks — unstandardized whole-brain betas
  make LinearSVC converge too slowly to refit `n_permutations + 1` times per mask (a one-off
  fit in `run_decoding.py` is fine; 101 refits on the same features is not), so the raw true
  accuracy may differ slightly from `run_decoding.py`'s number, though the true-vs-null
  comparison itself stays apples-to-apples. Also needs `--visual-cortex-mask`.

All decoders use `LeaveOneGroupOut` over the three runs (no temporal leakage).

## Results notebooks

| Notebook | Purpose |
|----------|---------|
| `searchlight_results.ipynb` | Aggregate category searchlight maps (vs empirical null) |
| `searchlight_v1_vs_v2_comparison.ipynb` | Compare searchlight variants |
| `decoding_results.ipynb` | Category WB/ROI accuracy + confusions across subjects |
| `frem_results.ipynb` | Category FREM weight maps + per-class AUC |
| `betas_qc_decoding.ipynb` | Beta-version QC: B→D category accuracy per subject + group, wholebrain vs visual-cortex |
| `label_shuffle_qc.ipynb` | Label-shuffle QC: true vs shuffled-label null distribution, empirical p-values per subject |
| `glmsingle_qc.ipynb` | GLMsingle fit QC (R², HRF, reliability) |
| `inspect_gm_masks.ipynb` | Inspect grey-matter / brain masks |
| RSA: `RSA_sandbox.ipynb`, `RSA_second_lvl.ipynb`, `dev_nilearn_firstlvl_rsa.ipynb` | Representational similarity analyses |
| Dev/sandbox: `dev_glmsingle_stim_cat.ipynb`, `dev_mvpa.ipynb`, `single_trial_GLM_sandbox.ipynb` | Prototyping (source of the `run_*.py` scripts) |

## `references/`

Collaborator single-trial / GLMsingle code kept for context, **not wired into the pipeline**
(SPM-LSS, GLMdenoise, and GLMsingle variants from other projects). See `references/README.md`.
