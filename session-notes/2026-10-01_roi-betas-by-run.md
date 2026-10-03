# Session log — ROI betas by run across the re-run models (2026-10-01)

**Companion notes:** [2026-09-25_concat-model-checks.md](2026-09-25_concat-model-checks.md) (findings 5–12: test-session vmPFC effect, H vs time).

Started as prep for the update meeting with Stephan and Viktor (talking points drawn from the vault note "LH - Supervisor update 2026-09", unchanged since 2026-09-28). Turned into a per-run ROI check of the vmPFC test-session effect across models, and a plan for a results-aggregation notebook.

---

## Findings

### 1. vmPFC: every chosen-value regressor dips in learning 2 and rises in test, so pooling cancels them out
Whole Bartra vmPFC mask, per-subject mean, N = 59. Learning 1 / learning 2 / test: GLM2 chosen Q −0.01 / −0.23 (p .051) / +0.53 (p < .001); GLM3 choice variable +0.00 / −0.24 (p .020) / +0.50 (p < .001); model-free chosen value +0.05 / −0.23 (p .016) / +0.52 (p < .001); GLM1 second-stimulus Q same shape, weaker (test +0.24, p .016); GLM1 first-stimulus Q flat. Pooled over runs, nothing reaches p < .05 (GLM2 chosen p .129, MF value p .053). So the learning-2 dip seen in the 2026-09-25 note (finding 7) holds for all three chosen-value regressors, not just one model, and it shows up at whole-ROI level. `roi_betas_by_session_stats.csv` (see Data produced), rows with roi = vmPFC. *Backing scripts removed from the repo, deferred to the aggregation notebook (open thread 3).*

### 2. Striatum and DLS/M1 at whole-mask level
Bartra striatum: Q is positive in test (GLM2 chosen +0.34, p < .001; MF value p .017) and GLM2 chosen Q is also positive in learning 1 (p .001). H in learning 1 is strongly negative in every per-session model (−0.7 to −1.3), consistent with the session-1 VIF. That's the collinear pair, not a result. In DLS (Guida) and M1, every H/frequency average is null in every run, as in 2026-09-25 finding 10: the DLS effect is focal, so whole-mask averages can't test it. Same stats CSV.

### 3. The pooled (`allruns`) contrast of a per-session model is the sum of the three runs, not their mean
Max |allruns − (Sn1 + Sn2 + Sn3)| = 1.4e-8 across subjects (GLM2 chosen Q, striatum). Per-session pooled betas are on 3× the per-run scale (and 3× a concat model's scale); divide by 3 before comparing them with per-run boxes. t-tests don't change.

```python
# Finding 3: pooled contrast = sum of per-run contrasts
import pandas as pd
r = pd.read_csv('/Users/hugofluhr/phd_local/data/LearningHabits/spm_outputs_cluster_reruns/aggregate/roi_betas_by_session.csv')
g = r[(r.model == 'GLM2 chosen') & (r.roi == 'striatum (Bartra)') & (r.contrast == 'second_stimxqval_chosen')].pivot(index='sub', columns='session', values='value')
print((g['allruns'] - g[['session-01', 'session-02', 'session-03']].sum(1)).abs().max())
```

### 4. No concat model has a single H regressor together with a time regressor
The cluster outputs have `glm2_all_runs_concat`, `glm2_chosen_all_runs_concat`, `…_hsplit` and `…_hsplit_time` (within-run trial order only). A within-run time regressor added to the unsplit model would only compete with within-run H. The informative test is unsplit H plus experiment-wide trial number, which is likely to be very collinear with H (run-level H rises about linearly with run order). Check its VIF from the bbt before spending a cluster run. Pointer added to open thread 2 of the 2026-09-25 note.

---

## Code shipped

None. The extraction (`scripts/roi_betas_by_session.py`) and plotting (`scripts/plot_roi_betas_by_session.py`) scripts written this session were removed from the repo on 2026-10-03, to be rebuilt in the aggregation notebook (open thread 3). The last copy of the extraction script is in cluster scratch: `/scratch/hfluhr/roi_betas/roi_betas_by_session.py` (not durable).

## Data produced

- Local: `~/phd_local/data/LearningHabits/spm_outputs_cluster_reruns/aggregate/` holds `roi_betas_by_session.csv` (7965 rows; model, contrast, session, roi, sub, value; n = 59 in every cell), `roi_betas_by_session_stats.csv` (mean/t/p, pooled per-session contrasts ÷ 3) and `roi_betas_by_session.png`. These back findings 1–3 until the notebook regenerates them.

## Git state at session end

Branch `model-reruns`, last commit `1f6d7d4` (local = origin). This note and the edit to the 2026-09-25 note are staged, not committed.

## Open threads

1. Rebuild the per-run ROI extraction and figure inside the aggregation notebook (thread 3); the old figure's legend overlapped the subtitle, so consider separate value and H figures.
2. Run unsplit chosen H + experiment-wide trial number in the concat design, after checking its VIF from the bbt (finding 4).
3. Results-aggregation notebook `notebooks/results/rerun_results_overview.ipynb`. Plan from this session: a shared model registry in `utils/rerun_models.py`; cluster scripts write CSVs to `spm_outputs/aggregate_<date>/`; sections cover sample/VIF, the second-level results matrix from the `compare_second_lvl_…/svc/` peak tables plus the claims check, ROI betas by run, leave-one-subject-out focal ROIs, H vs time, model t-map similarity, and a findings summary. Open decisions: name, refactor, univariate-only scope.
4. vmPFC learning-2 dip (finding 1): why is learning 2 negative? Unresolved (2026-09-25 open thread 4).
5. Meeting follow-ups from Maria Eckstein (vault, 2026-09-30): ask Stephan/Viktor about Palminteri context model and forgetting on Q/H.
