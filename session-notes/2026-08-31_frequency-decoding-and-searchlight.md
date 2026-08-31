# Session log — Frequency decoding (ROI + searchlight)

**Date:** 2026-08-31  
**Companion note:** [2026-08-30_rsa-confound-controls-and-validation.md](2026-08-30_rsa-confound-controls-and-validation.md)

Followed up on the robust RSA choice-frequency effect (VC β=+0.337, fusiform β=+0.373,
FDR q<0.001) with complementary decoding analyses: (1) ROI-level binary classification
of the ±1 frequency label, (2) whole-brain searchlight classification for voxel-level
localization. Searchlight decoding was chosen over searchlight RSA — no RSA searchlight
infrastructure existed, and per-sphere 5-predictor regression on 28 data points would be
noisy.

---

## Findings

### 1. ROI classification: VC 74.4%, fusiform 67.2%, parietal 52.9% (all FDR q<0.001)

Full n=58 results. Whole brain 62.0% (t=14.9). Parietal significant — not in RSA.
All subcortical ROIs at chance (putamen 49.9%, habit 50.6%, vmPFC 50.7%, striatum 50.8%).
Derivation: `frequency_decoding_results.ipynb` §2.

### 2. Searchlight: 8,245 voxels decode frequency above chance (FDR q<0.05)

16% of brain voxels. Almost exclusively above-chance (53 below). Max t=17.6 in right
fusiform (42, −57, −15). One large occipitotemporal cluster (232k mm³) plus smaller
frontal clusters (bilateral PFC, t=3.5–5.5). Spatial correlation with category searchlight
r=0.868. Derivation: `frequency_searchlight_results.ipynb` §3.

### 3. Searchlight ROI means reveal effects beyond RSA

Mean searchlight accuracy within each RSA ROI mask (FDR across 8 ROIs):
- Fusiform: 58.1%***, Visual cortex: 53.8%*** — consistent with RSA
- Parietal: 50.7%***, Premotor: 50.3%** — **NEW**: not significant in ROI RSA
- vmPFC: 50.5%*, Striatum: 50.5%* — marginal in RSA, now significant
- Putamen: 50.1%, Habit: 50.2% — null in both analyses

Searchlight is more sensitive for weak distributed effects because it tests local
neighborhoods (~30–90 voxels) rather than full anatomical masks. Derivation:
`frequency_searchlight_results.ipynb` §5.

### 4. Demeaning variant divergence is variance reduction, not confound

Run×cat demeaning drops accuracy ~13% in VC (t=11.2), ~8% in fusiform (t=8.5), both
p<0.001. But run_cat_demeaned is still well above chance (VC 61.7%, fusiform 58.8%).
Drop is limited to visual regions and absent in subcortical ROIs. On the non-figure
subset, category is perfectly balanced with frequency by design (one +1 and one −1 per
category), so the drop reflects removed pattern variance, not a confound. Derivation:
`frequency_decoding_results.ipynb` §2 paired-difference table.

---

## Code shipped

| File | Change | Git state |
|------|--------|-----------|
| `multivariate/run_frequency_decoding.py` | New: ROI-level ±1 frequency classification | rsa-roi (772a837) |
| `multivariate/submit_frequency_decoding.sh` | New: SLURM submit, 1h walltime | rsa-roi (f9d9ecf) |
| `multivariate/run_frequency_searchlight.py` | New: whole-brain searchlight frequency classification | rsa-roi (772a837) |
| `multivariate/submit_frequency_searchlight.sh` | New: SLURM array submit | rsa-roi (772a837) |
| `multivariate/frequency_searchlight_results.ipynb` | New: executed with full n=58 | rsa-roi |
| `multivariate/frequency_decoding_results.ipynb` | New: executed with full n=58 | rsa-roi |

## Data produced

- **Searchlight (job 5413066):** 58/58 subjects complete. Results at
  `derivatives/searchlight/sub-*/sub-*_searchlight_frequency.nii.gz`. Synced locally.
- **ROI decoding (jobs 5413131 + 5423107):** 58/58 subjects complete. Results at
  `derivatives/frequency_decoding/sub-*/sub-*_frequency_decoding.csv`. Synced locally.
  Job 5413131 timed out at 20 min (whole-brain LinearSVC convergence); 5423107 finished
  the remaining 38 subjects in 40 min with 1h walltime.

## Git state

Branch: `rsa-roi`. Will push after final commit.

---

## Open threads

1. ~~ROI decoding job~~ → **DONE** (58/58 complete, finding 1 above).
2. ~~Demeaning variant divergence~~ → **EXPLAINED** (finding 4 above).
3. **Frontal searchlight clusters** — small but FDR-surviving clusters in bilateral PFC
   (t=3.5–5.5). Anatomical labeling and interpretation needed.
4. **Searchlight–RSA correlation** — r=0.868 with category searchlight. Are there
   frequency-specific voxels outside the category-sensitive territory?
5. **RSA–decoding subject-level convergence** — scatter plots in
   `frequency_decoding_results.ipynb` §5 need interpretation.
6. From prior session: FDR-correct §8c interaction; interpret VC choice_rate negative
   effect; vmPFC/striatum marginal; per-run interaction; branch not merged to main.
