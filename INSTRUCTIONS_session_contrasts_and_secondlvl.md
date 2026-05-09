# Instructions: Add Session Contrasts → Export → Second-Level Analysis

These instructions apply to **three GLM models**. Paths marked with `<PLACEHOLDER>` must be filled in before running.

---

## Logging

For each of the three steps, capture MATLAB console output to a log file so there is a record of what ran and any warnings. Use a timestamped filename and store logs alongside the relevant output directory, for example:

```bash
matlab -nodisplay -r "run('add_session_contrasts_glm2.m'); exit" \
  | tee /path/to/logs/add_session_contrasts_$(date +%Y%m%d_%H%M%S).log
```

At minimum the log should capture: which subjects were processed vs skipped, how many contrasts were added/found/excluded, any warnings, and the final "done" confirmation for each step. If running MATLAB interactively rather than from the command line, redirect diary output at the top of each script run (`diary /path/to/logfile.txt`).

---

## Impact on existing first-level directories

**Step 1 modifies the first-level GLM directories** — this is by design and unavoidable. It appends new `con_XXXX.nii` images and updates `SPM.mat` with the new `xCon` entries. Crucially, it uses `delete=0` (append mode) and skips any subject that already has session contrasts, so existing contrast images and model estimates are never overwritten or removed.

**Steps 2 and 3 are read-only with respect to the first-level directories.** The export script only reads from them and writes everything to the new output directories. The second-level script reads only from those export directories.

---

## Paths to fill in

| Variable | Description | Value |
|----------|-------------|-------|
| `GLM_ALLRUNS` | Path to glm2-all-runs directory | `<PATH_TO_GLM2_ALL_RUNS>` |
| `GLM_CHOSEN_ALLRUNS` | Path to glm2-chosen-all-runs directory | `<PATH_TO_GLM2_CHOSEN_ALL_RUNS>` |
| `GLM_THIRD` | Path to the third GLM directory | `<PATH_TO_THIRD_GLM>` |
| `EXPORT_OUT_ALLRUNS` | New export output directory for glm2-all-runs | `<PATH_TO_EXPORT_OUTPUT_GLM2_ALL_RUNS>` |
| `EXPORT_OUT_CHOSEN_ALLRUNS` | New export output directory for glm2-chosen-all-runs | `<PATH_TO_EXPORT_OUTPUT_GLM2_CHOSEN_ALL_RUNS>` |
| `EXPORT_OUT_THIRD` | New export output directory for the third model | `<PATH_TO_EXPORT_OUTPUT_THIRD_GLM>` |
| `SPM_PATH` | Path to SPM12 installation | `<PATH_TO_SPM12>` |

---

## Relevant scripts

All scripts are in the repository at:
`learning-habits-analysis/matlab/`

- **Session contrast addition**: `matlab/first_lvl/add_session_contrasts_glm2.m`
- **Export (with session separation)**: `matlab/export_first_lvl_contrasts_with_sessions.m`
- **Second-level analysis**: `matlab/second_lvl/second_lvl.m`

---

## Step 1 — Add per-session contrasts to each GLM

**Script**: `matlab/first_lvl/add_session_contrasts_glm2.m`

This script appends per-session t-contrasts to already-estimated SPM.mat files. It is safe to re-run: it will skip any subject that already has session contrasts. It adds one contrast per session (Session 1 = learning1, Session 2 = learning2, Session 3 = test) for each contrast in `connames`.

Run it **three times**, once per GLM. The only variable to change between runs is `glm_root` at the top of the script:

```matlab
% Run 1: glm2-all-runs
glm_root = '<PATH_TO_GLM2_ALL_RUNS>';

% Run 2: glm2-chosen-all-runs
glm_root = '<PATH_TO_GLM2_CHOSEN_ALL_RUNS>';

% Run 3: third model
glm_root = '<PATH_TO_THIRD_GLM>';
```

**Verify after each run**: check the MATLAB console output. Each subject line should read `[DONE] sub-XX: added N per-session contrasts.` Any `[SKIP]` messages are expected only if a subject already had session contrasts or lacked an expected regressor.

---

## Step 2 — Export contrasts to new output directories

**Script**: `matlab/export_first_lvl_contrasts_with_sessions.m`

This is a MATLAB function. It exports both all-runs average contrasts and per-session contrasts into a structured output directory, keeping existing and new contrasts together. The output structure will be:

```
<EXPORT_OUT>/
  allruns/
    contrast-01_<name>/   ← one dir per contrast
    contrast-02_<name>/
    ...
    contrast_list_order_allruns.txt
    contrasts_manifest.tsv
  session-01/
    contrast-01_<name>/
    ...
    contrast_list_order_session-01.txt
  session-02/
    ...
  session-03/
    ...
  contrasts_manifest_sessions.tsv
```

Run it **three times**, once per GLM. Use `copy = true` so the export directory is self-contained (no symlinks that break if source moves):

```matlab
addpath('<PATH_TO_SPM12>');
spm('Defaults', 'fMRI'); spm_jobman('initcfg');

% Run 1: glm2-all-runs
export_first_lvl_contrasts_with_sessions( ...
    '<PATH_TO_GLM2_ALL_RUNS>', ...
    '<PATH_TO_EXPORT_OUTPUT_GLM2_ALL_RUNS>', ...
    'copy', true);

% Run 2: glm2-chosen-all-runs
export_first_lvl_contrasts_with_sessions( ...
    '<PATH_TO_GLM2_CHOSEN_ALL_RUNS>', ...
    '<PATH_TO_EXPORT_OUTPUT_GLM2_CHOSEN_ALL_RUNS>', ...
    'copy', true);

% Run 3: third model
export_first_lvl_contrasts_with_sessions( ...
    '<PATH_TO_THIRD_GLM>', ...
    '<PATH_TO_EXPORT_OUTPUT_THIRD_GLM>', ...
    'copy', true);
```

**Verify**: each output directory should contain an `allruns/` subdirectory with `contrast_list_order_allruns.txt` and session subdirectories `session-01/` through `session-03/`, each containing their own `contrast_list_order_session-0N.txt`.

---

## Step 3 — Run second-level analysis

**Script**: `matlab/second_lvl/second_lvl_all_runs.m`

This script reads `contrast_list_order_allruns.txt` from `first_lvl_dir`, then for each contrast looks for a `contrast-NN_<name>/` subdirectory containing exported `.nii` files and runs a one-sample t-test.

The script loops over `allruns/`, `session-01/`, `session-02/`, `session-03/` automatically, skipping any that don't exist. Output is written to `<EXPORT_OUT>/second-lvl/<subdir>/` with a `subjects_included.txt` manifest per subdir.

Review the exclusion list at the top of the script and confirm it is correct for these models:

```matlab
excluded_subjects = {'sub-44', 'sub-48', 'sub-68', 'sub-17', 'sub-31'};
```

Run it **three times**, setting `export_root` to each export output directory (the root, not `allruns/`):

```matlab
% Run 1: glm2-all-runs
export_root = '<PATH_TO_EXPORT_OUTPUT_GLM2_ALL_RUNS>';

% Run 2: glm2-chosen-all-runs
export_root = '<PATH_TO_EXPORT_OUTPUT_GLM2_CHOSEN_ALL_RUNS>';

% Run 3: third model
export_root = '<PATH_TO_EXPORT_OUTPUT_THIRD_GLM>';
```

**Verify**: `<EXPORT_OUT>/second-lvl/` should appear after each run, containing subdirectories `allruns/`, `session-01/`, `session-02/`, `session-03/`. Each contrast subdirectory inside those should contain `SPM.mat`, `beta_0001.nii`, `con_0001.nii`, `spmT_0001.nii`, and a `subjects_included.txt` manifest.

---

## Suggested order and debugging notes

1. Complete Step 1 for all three models before proceeding, so all SPM.mat files are fully updated.
2. After Step 1, spot-check one or two subjects per model: load their SPM.mat in MATLAB and confirm `SPM.xCon` contains contrasts named e.g. `first_stim - Session 1`.
3. After Step 2, check the `contrasts_manifest.tsv` files to confirm subject counts are as expected.
4. If the export script throws a contrast mismatch error, a subject's SPM.mat likely has a different contrast order — investigate that subject's first-level run before proceeding.
5. If second-level finds zero contrast files for a given contrast, double-check that the sanitized contrast name in the order file matches the directory names created in Step 2 (the `sanitize` function lowercases and replaces spaces with hyphens, stripping special characters).
