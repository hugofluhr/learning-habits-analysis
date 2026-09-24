# Session log: tier-1 re-runs, second-level comparison with the manuscript, concat models

**Date:** 2026-09-24
**Companion notes:** [2026-09-24_spm-pipeline-merge-and-reruns.md](2026-09-24_spm-pipeline-merge-and-reruns.md) (same day, earlier: PR #2 merge, Phase B, `model-reruns` started), [2026-09-24_glm-concat-validation.md](2026-09-24_glm-concat-validation.md) (concat design on sub-01)
**Report (artifact):** https://claude.ai/artifact/Bf6dG3KhPB4oGrAnmT6RvX, built from `report.json` + the tables below (the HTML template itself lives only in the artifact)

Ran all tier-1 models plus the GLM1 concat model on the full cohort, then built a tool to compare the re-run second levels with the old runs and with the results stated in the manuscript and vault. It turned into a VIF-centred reading: GLM1 and GLM2 chosen H-value results reproduce but are not interpretable, and the concat design is the candidate fix. Ended with a GLM2 chosen concat model, which fixes the VIF and gives a stronger bilateral DLS H-value effect (finding 8).

---

## Findings

### 1. All tier-1 models and the GLM1 concat re-ran on the cluster (tag `2026-09-24-14-56`)
`glm2_all_runs`, `glm2_chosen_all_runs`, `glm3_chosen_choice_var`, `glm2_mf_chosenval`, `glm2_mf_chosenfrequ`, `glm2_all_runs_concat`: 60 first levels each (62 bbt subjects minus sub-04/45), N = 59 at second level in both old and new runs, identical subject lists. First levels took ~5 min per model as SLURM arrays; downstream ~6 min.

### 2. `svc_report.m` reproduces the manuscript's SVC numbers exactly, and the manuscript's p-values are cluster-level FWE
On the old GLM1 run: t = 4.19 / 4.12 / 4.23 at the manuscript's coordinates, with cluster p(FWE) = .007 / .016 / .030 = the manuscript's p-values (peak p(FWE) = .004 / .006 / .021). `spm_VOI` itself fails in `matlab -batch` (`spm_list('List')` needs the results window), so `svc_report.m` repeats its mask branch and calls `spm_list('Table')`. Tables: `spm_outputs/compare_second_lvl_2026-09-24-14-56/svc/<run>.csv`.

### 3. The re-run reproduces the old second levels; 16 of 17 manuscript statements hold on both
The failing one fails on the old run too: GLM1 "no H-value effect in motor ROIs" is contradicted by an M1 cluster (full HMAT motor mask cluster p .021 old / .019 new; M1-only mask .007 both). Modulator t-maps old vs new r = 0.998–1.000. Only vault-claim change: GLM2 chosen H-value in the premotor-only mask, cluster p .026 → .063 (peak p .023 → .011). `bash scripts/submit_compare_second_lvl.sh 2026-09-24-14-56` → `report.md` / `report.json`.

### 4. Bug A's footprint is the test session's main effects
In the model-free models' per-session second levels, sessions 1–2 are identical old vs new (r = 1.000) and session 3 changes: purple_frame r = −0.10, response 0.16–0.18, first_stim 0.79 (`report.json`, `tmap_corr`). All-runs main effects r = 0.94–0.998.

### 5. Session-1 H-value VIF: GLM1 and GLM2 chosen only; concat fixes it
Median (max) H-pmod VIF in Sn1: GLM1 12.3–14.0 (26–27), GLM2 chosen 17.6 (29.2), above 5 in all 60 subjects; Sn2/Sn3 ≤ 3.7. Choice variable, model-free value/frequency ≤ 1.4 median. GLM1 concat 2.4–2.5 (max 3.7). So the manuscript's H-value results (GLM1/GLM2 chosen all-runs) reproduce but are not interpretable; their Q-value results are fine. Reports: `spm_outputs/<model>_2026-09-24-14-56/vif_report/`; medians computed with the first block below.

### 6. H-value evidence without the VIF problem
- GLM1 concat, H sum: right DLS/Guida (27, 9, 6) k 11 cluster p .021; M1 (18, −30, 69) k 59 p < .001; vmPFC (−3, 42, −15) k 11 p .004; second-stimulus H alone gives smaller clusters in the same places. Caveat: pooled H keeps run-level offsets.
- Sessions 2+3 only: GLM1 second-stim H has an M1 cluster (30, −33, 66) k 30, p .005 (M1 mask); nothing in DLS. GLM2 chosen: one whole-brain cluster (−9, 15, 20) k 49 p .016, nothing in ROIs at cluster level.
Tables: second block below.

### 7. Q-values in the GLM1 concat vs GLM1
Q sum in VS unchanged (bilateral, t 4.7, cluster p .004/.006). Preregistered first-stimulus Q shrinks from two VS clusters (t ≈ 4.2, p .011/.017) to one voxel (t 3.3, p .036), and its Guida clusters disappear. vmPFC: nothing in either. Not yet explained. Same second block below, with `'qval'`.

### 8. GLM2 chosen concat fixes the VIF and strengthens the bilateral DLS H-value effect (major)
`glm2_chosen_all_runs_concat_scrubbed_2026-09-24-17-40` (N = 59): H-chosen VIF median 2.5, max 3.3, 0/60 above 5 (per-session 17.6, all > 10). Chosen H survives SVC in DLS bilaterally, right (27, 9, 6) t 6.29 k 38 cluster p .001 (also whole-brain FWE, peak p .001) and left (−27, 9, 2) t 5.24 k 27 p .003, both stronger than per-session (t 4.82 / 4.47); plus M1 (18, −30, 72) k 54 p .001 and vmPFC (−3, 42, −15) k 18 p .002. The manuscript's motor peak (48, −6, 20) is gone. Chosen Q: right VS unchanged (9, 9, −4; t 4.87, p .007), left VS drops from 18 voxels (p .002) to one voxel (p .035), the same pattern as the GLM1 concat (finding 7). Second code block below with `'glm2_chosen_all_runs_concat_scrubbed_2026-09-24-17-40'`; VIF with the first block (tag `2026-09-24-17-40`).

### 9. Tier 2 all share the VIF problem, by construction for #8/#9
Within-session re-z-scoring is an affine map per session, absorbed by the unmodulated onset regressor, so per-session VIFs of `*_run_zscore` equal GLM1/GLM2 chosen; `glm2_merged_stim` was shown high in June. Tier 2 put on hold by Hugo.

```python
# Median / max pmod VIF per session and model (finding 5). Run from the repo root on the cluster.
import glob, os, pandas as pd
for f in sorted(glob.glob('/home/hfluhr/data/learninghabits/spm_outputs/*_2026-09-24-1[47]-[45]*/vif_report/tables/vifs_per_subject_session.csv')):
    d = pd.read_csv(f); pm = [c for c in d.columns if '^1' in c]
    print(f.split('/')[-4]); print(d.groupby('session')[pm].agg(['median', 'max']).round(1).to_string())
```

```python
# Significant clusters (either level) for H or Q contrasts, one row per cluster at its strongest peak (findings 6-7).
import pandas as pd
def clusters(run, pattern, prefix='allruns/'):
    d = pd.read_csv(f'/home/hfluhr/data/learninghabits/spm_outputs/compare_second_lvl_2026-09-24-14-56/svc/{run}.csv').dropna(subset=['peak_t'])
    d = d[d.model.str.startswith(prefix) & d.model.str.contains(pattern)]
    d = d.drop_duplicates(['model', 'region', 'cluster_k', 'cluster_p_fwe'])
    return d[(d.cluster_p_fwe < .05) | (d.peak_p_fwe < .05)][['model', 'region', 'cluster_k', 'peak_t', 'x', 'y', 'z', 'cluster_p_fwe', 'peak_p_fwe']]
print(clusters('glm2_all_runs_concat_scrubbed_2026-09-24-14-56', 'hval'))
print(clusters('glm2_all_runs_scrubbed_2026-09-24-14-56', 'hval', prefix='session-02-03/'))
print(clusters('glm2_all_runs_scrubbed_2026-09-24-14-56', 'qval'))
print(clusters('glm2_chosen_all_runs_concat_scrubbed_2026-09-24-17-40', 'hval'))
print(clusters('glm2_chosen_all_runs_concat_scrubbed_2026-09-24-17-40', 'qval'))
print(clusters('glm2_chosen_all_runs_scrubbed_2026-09-24-14-56', 'qval'))
```

---

## Code shipped (branch `model-reruns`, pushed)

| Commit | What |
|---|---|
| `0956c99` | Tier-1 scripts ported with the Bug A fix (`glm3_chosen_choice_var`, `glm2_mf_val`, `glm2_mf_frequ`) |
| `46c273e` | Export fix: models without session contrasts (`unique([])` is 0×1) |
| `044f149` | `svc_report.m`, `compare_second_lvl.py`, `second_lvl_claims.csv`, `submit_compare_second_lvl.sh` + job file |
| `54039f8` | `glm2_chosen_all_runs_concat.m` (same changes as GLM1 concat; keeps the resp/nresp split of second_stim) |

## Data produced (cluster)
- `spm_format/outputs/<model>_2026-09-24-14-56/` (6 first levels) and `spm_outputs/<model>_2026-09-24-14-56/` (exports, second levels, `vif_report/`).
- `spm_format/reference/second_lvl/`: the old second levels (6 runs, 63 models), copied from the laptop's `spm_outputs_noSDC/`.
- `spm_outputs/compare_second_lvl_2026-09-24-14-56/`: `svc/*.csv`, `report.md`, `report.json`.
- GLM2 chosen concat, tag `2026-09-24-17-40`: `spm_format/outputs/glm2_chosen_all_runs_concat_scrubbed_2026-09-24-17-40/` and `spm_outputs/…` (export, second level, `vif_report/`); SVC table in `compare_second_lvl_2026-09-24-14-56/svc/`. Jobs 6456476 → 6456477 → 6456478, all completed.
- Scratch to delete: `spm_format/reference/compare_test_code/` (pre-commit copy of the code), `spm_format/reference/voi_test*`, `svc_test*`.

## Git state at session end
`model-reruns` = `origin/model-reruns` = `54039f8` (on top of `main` `7124d0a`); the cluster checkout is on `model-reruns`. Not merged.

## Open threads
1. **Bulletproof the GLM2 chosen concat model and understand why part of the Q-value result is lost** (left VS here, first-stimulus VS in the GLM1 concat). Start with the run-level offsets in the pooled H/Q regressors (between-run variance share per subject; a per-run z-scored variant as control), and check whether the run constants absorb Q signal.
2. vmPFC appears for H in both concat models but in no per-session model: part of thread 1.
3. Manuscript: GLM1 "no motor effect" sentence is wrong at cluster level in both runs; the H-value sections depend on the VIF decision.
4. Tier 2 and `glm2_all_runs_diff_timing` on hold (all share the Sn1 VIF).
5. Second levels test positive effects only; the vault's negative Sn2+Sn3 cluster is not covered.
