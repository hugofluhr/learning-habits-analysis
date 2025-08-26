%% batch_vif_across_runs.m
% Compute VIFs across subjects/runs and summarize.
% Layout expected: base_dir/sub-XX/run-X/SPM.mat
%
% Outputs:
%   - vifs_per_run.csv  (all runs)
%   - vif_summary.csv   (aggregated per regressor)
%
% Requirements:
%   - spm_vif.m on path (the helper from our previous message)
%   - SPM.mat present in each run folder
%
% Notes:
%   - "Classic VIF" -> via spm_vif, excluding constants/DCT/motion (NOT removing R1/R12)
%   - "Conditional VIF" -> in-script: regress out R-confounds (R^digits), then VIF on task cols
%   - Regressor names are normalized by removing 'Sn(#) ' prefixes for aggregation.

clear; clc;

%% ==== CONFIG ====
base_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/spm_outputs/glm1_2025-06-05-05-15';

% Patterns considered nuisance (excluded from VIF sets)
pat_constants = 'constant';   % SPM's per-session intercepts include "constant"

% Confound regex for conditional VIF projection (R1, R12, ...):
confound_regex = 'R\d+';

% VIF thresholds for counts
thr5  = 5;
thr10 = 10;

% Output files
per_run_csv = fullfile(base_dir, 'vifs_per_run.csv');
summary_csv = fullfile(base_dir, 'vif_summary.csv');

%% ==== HELPERS ====

normalize_name = @(s) regexprep(s, '^Sn\(\d+\)\s*', '');  % drop SPM session prefix e.g., 'Sn(1) first_stim'

% Compute VIF table for a given design matrix (columns already selected)
function T = local_vif_table(Xin, names)
    p = size(Xin,2);
    VIF = nan(p,1); R2 = nan(p,1); tol = nan(p,1);

    % Remove zero-variance columns (avoid NaNs)
    keep = var(Xin) > eps;
    Xin  = Xin(:, keep);
    names = names(keep);
    p = size(Xin,2);

    for j = 1:p
        y  = Xin(:, j);
        Xo = Xin(:, setdiff(1:p, j));
        if isempty(Xo) || var(y) < eps
            VIF(j) = 1; R2(j) = 0; tol(j) = 1;
            continue;
        end
        beta = Xo \ y;
        yhat = Xo * beta;
        TSS = sum((y - mean(y)).^2);
        RSS = sum((y - yhat).^2);
        r2  = max(0, min(1, 1 - RSS/TSS));
        R2(j)  = r2;
        tol(j) = max(eps, 1 - r2);
        VIF(j) = 1 ./ tol(j);
    end

    T = table(string(names(:)), R2(:), tol(:), VIF(:), ...
        'VariableNames', {'Name','R2','Tolerance','VIF'});
end

function tf = any_ci_contains(strs, patterns)
    % case-insensitive "contains any"
    low = lower(string(strs));
    tf = false(size(low));
    for p = string(patterns)
        tf = tf | contains(low, lower(p));
    end
end

%% ==== SCAN DATASET ====
subs = dir(fullfile(base_dir, 'sub-*'));
per_run_rows = [];   % will hold tables vertically concatenated

for si = 1:numel(subs)
    if ~subs(si).isdir, continue; end
    sub_id = subs(si).name;

    runs = dir(fullfile(base_dir, sub_id, 'run-*'));
    for ri = 1:numel(runs)
        if ~runs(ri).isdir, continue; end
        run_id = runs(ri).name;

        spm_mat = fullfile(base_dir, sub_id, run_id, 'SPM.mat');
        if ~exist(spm_mat, 'file')
            fprintf('Skipping (no SPM.mat): %s/%s\n', sub_id, run_id);
            continue;
        end

        % Load SPM and extract design
        S = load(spm_mat, 'SPM');
        SPM = S.SPM;
        X = SPM.xX.X;                    % [timepoints x regressors]
        names = string(SPM.xX.name(:));  % column names

        % Identify column categories
        is_const  = any_ci_contains(names, {pat_constants});

        % Confounds for projection: regex ^R\d+
        is_conf = false(size(names));
        for k = 1:numel(names)
            is_conf(k) = ~isempty(regexp(names(k), confound_regex, 'once'));
        end

        % "Task" set for VIFs = not (constants/confounds)
        is_task = ~(is_const | is_conf);

        % --- (1) Classic VIF using spm_vif (task-only, but NOT removing R* there) ---
        try
            T_classic = spm_vif(spm_mat, ...
                'excludePatterns', {pat_constants}, ...
                'display', false);
            % Keep only rows that correspond to our is_task mask after name normalization
            % (spm_vif already excluded constants/DCT/motion/outliers; it may still include R*)
            % To align, filter by names present in current run AND marked as is_task:
            keep_names = names(is_task);
            T_classic = T_classic(ismember(string(T_classic.Name), keep_names), :);
            T_classic.Properties.VariableNames{'VIF'} = 'VIF_classic';
        catch ME
            warning('spm_vif failed in %s/%s: %s', sub_id, run_id, ME.message);
            % Fallback: compute classic locally on is_task set
            Ttmp = local_vif_table(X(:, is_task), names(is_task));
            Ttmp.Properties.VariableNames{'VIF'} = 'VIF_classic';
            T_classic = Ttmp;
        end

        % --- (2) Conditional VIF: project out confounds, then VIF on task columns ---
        C = X(:, is_conf);
        n = size(X,1);
        if isempty(C)
            X_task_cond = X(:, is_task);
        else
            P = C * pinv(C);           % projector onto confound space
            M = eye(n) - P;            % residual-maker
            X_task_cond = M * X(:, is_task);
        end
        T_cond = local_vif_table(X_task_cond, names(is_task));
        T_cond.Properties.VariableNames{'VIF'} = 'VIF_conditional';

        % --- Merge the two VIF columns on Name ---
        Tm = outerjoin(T_classic(:,{'Name','VIF_classic'}), ...
                       T_cond(:,   {'Name','VIF_conditional'}), ...
                       'Keys','Name','MergeKeys',true);
        % Normalize names for aggregation across runs (strip Sn(#))
        Tm.NormName = arrayfun(@(s) normalize_name(char(s)), Tm.Name, 'UniformOutput', false);

        % Append identifiers
        Tm.Subject = repmat(string(sub_id), height(Tm), 1);
        Tm.Run     = repmat(string(run_id), height(Tm), 1);

        % Reorder columns
        Tm = movevars(Tm, {'Subject','Run','Name','NormName'}, 'Before', 1);

        % Collect
        per_run_rows = [per_run_rows; Tm]; %#ok<AGROW>
    end
end

if isempty(per_run_rows)
    error('No runs processed. Check base_dir and folder layout.');
end

% Convert to table (if it isn't already)
if ~istable(per_run_rows)
    per_run_rows = struct2table(per_run_rows);
end

% Write per-run CSV
writetable(per_run_rows, per_run_csv);
fprintf('Wrote per-run VIFs: %s\n', per_run_csv);

%% ==== SUMMARY STATS ACROSS RUNS ====
% Helper aggregators (ignore NaNs)
mean_no_nan = @(x) mean(x,'omitnan');
med_no_nan  = @(x) median(x,'omitnan');
max_no_nan  = @(x) max(x,[],'omitnan');
p95_no_nan  = @(x) prctile(x,95);
over5       = @(x) sum(x > thr5,  'omitnan');
over10      = @(x) sum(x > thr10, 'omitnan');
count_nonan = @(x) sum(~isnan(x));

%% (A) Overall regressor-wise summary (as before)
G_over = findgroups(string(per_run_rows.NormName));
reg_names_over = splitapply(@(s) string(s(1)), string(per_run_rows.NormName), G_over);

N_over        = splitapply(count_nonan, per_run_rows.VIF_classic, G_over);
mean_classic  = splitapply(mean_no_nan, per_run_rows.VIF_classic, G_over);
med_classic   = splitapply(med_no_nan,  per_run_rows.VIF_classic, G_over);
max_classic   = splitapply(max_no_nan,  per_run_rows.VIF_classic, G_over);
p95_classic   = splitapply(p95_no_nan,  per_run_rows.VIF_classic, G_over);
n5_classic    = splitapply(over5,       per_run_rows.VIF_classic, G_over);
n10_classic   = splitapply(over10,      per_run_rows.VIF_classic, G_over);

mean_cond     = splitapply(mean_no_nan, per_run_rows.VIF_conditional, G_over);
med_cond      = splitapply(med_no_nan,  per_run_rows.VIF_conditional, G_over);
max_cond      = splitapply(max_no_nan,  per_run_rows.VIF_conditional, G_over);
p95_cond      = splitapply(p95_no_nan,  per_run_rows.VIF_conditional, G_over);
n5_cond       = splitapply(over5,       per_run_rows.VIF_conditional, G_over);
n10_cond      = splitapply(over10,      per_run_rows.VIF_conditional, G_over);

summary_overall = table( ...
    repmat("Overall", numel(reg_names_over), 1), ... % Scope
    repmat("ALL",     numel(reg_names_over), 1), ... % Run
    reg_names_over, N_over, ...
    mean_classic, med_classic, p95_classic, max_classic, n5_classic, n10_classic, ...
    mean_cond,    med_cond,    p95_cond,    max_cond,    n5_cond,    n10_cond, ...
    'VariableNames', { ...
        'Scope','Run','Regressor','NRuns', ...
        'MeanVIF_classic','MedianVIF_classic','P95VIF_classic','MaxVIF_classic','CountVIFgt5_classic','CountVIFgt10_classic', ...
        'MeanVIF_cond','MedianVIF_cond','P95VIF_cond','MaxVIF_cond','CountVIFgt5_cond','CountVIFgt10_cond' ...
    });
summary_overall.PctVIFgt5_classic  = 100 * summary_overall.CountVIFgt5_classic  ./ summary_overall.NRuns;
summary_overall.PctVIFgt10_classic = 100 * summary_overall.CountVIFgt10_classic ./ summary_overall.NRuns;
summary_overall.PctVIFgt5_cond     = 100 * summary_overall.CountVIFgt5_cond     ./ summary_overall.NRuns;
summary_overall.PctVIFgt10_cond    = 100 * summary_overall.CountVIFgt10_cond    ./ summary_overall.NRuns;

%% (B) Same regressor-wise summary, but split by run label (run-1/run-2/run-3)
G_byrun = findgroups(string(per_run_rows.Run), string(per_run_rows.NormName));
run_labels   = splitapply(@(s) string(s(1)), string(per_run_rows.Run),     G_byrun);
reg_names_br = splitapply(@(s) string(s(1)), string(per_run_rows.NormName), G_byrun);

N_byrun        = splitapply(count_nonan, per_run_rows.VIF_classic, G_byrun);
br_mean_class  = splitapply(mean_no_nan, per_run_rows.VIF_classic, G_byrun);
br_med_class   = splitapply(med_no_nan,  per_run_rows.VIF_classic, G_byrun);
br_max_class   = splitapply(max_no_nan,  per_run_rows.VIF_classic, G_byrun);
br_p95_class   = splitapply(p95_no_nan,  per_run_rows.VIF_classic, G_byrun);
br_n5_class    = splitapply(over5,       per_run_rows.VIF_classic, G_byrun);
br_n10_class   = splitapply(over10,      per_run_rows.VIF_classic, G_byrun);

br_mean_cond   = splitapply(mean_no_nan, per_run_rows.VIF_conditional, G_byrun);
br_med_cond    = splitapply(med_no_nan,  per_run_rows.VIF_conditional, G_byrun);
br_max_cond    = splitapply(max_no_nan,  per_run_rows.VIF_conditional, G_byrun);
br_p95_cond    = splitapply(p95_no_nan,  per_run_rows.VIF_conditional, G_byrun);
br_n5_cond     = splitapply(over5,       per_run_rows.VIF_conditional, G_byrun);
br_n10_cond    = splitapply(over10,      per_run_rows.VIF_conditional, G_byrun);

summary_byrun = table( ...
    repmat("ByRun", numel(reg_names_br), 1), ... % Scope
    run_labels, reg_names_br, N_byrun, ...
    br_mean_class, br_med_class, br_p95_class, br_max_class, br_n5_class, br_n10_class, ...
    br_mean_cond,  br_med_cond,  br_p95_cond,  br_max_cond,  br_n5_cond,  br_n10_cond, ...
    'VariableNames', { ...
        'Scope','Run','Regressor','NRuns', ...
        'MeanVIF_classic','MedianVIF_classic','P95VIF_classic','MaxVIF_classic','CountVIFgt5_classic','CountVIFgt10_classic', ...
        'MeanVIF_cond','MedianVIF_cond','P95VIF_cond','MaxVIF_cond','CountVIFgt5_cond','CountVIFgt10_cond' ...
    });
summary_byrun.PctVIFgt5_classic  = 100 * summary_byrun.CountVIFgt5_classic  ./ summary_byrun.NRuns;
summary_byrun.PctVIFgt10_classic = 100 * summary_byrun.CountVIFgt10_classic ./ summary_byrun.NRuns;
summary_byrun.PctVIFgt5_cond     = 100 * summary_byrun.CountVIFgt5_cond     ./ summary_byrun.NRuns;
summary_byrun.PctVIFgt10_cond    = 100 * summary_byrun.CountVIFgt10_cond    ./ summary_byrun.NRuns;

%% Combine and write (single CSV)
summary_all = [summary_overall; summary_byrun];

% Sort: show ByRun blocks grouped by run, worst (MaxVIF_cond) first; then Overall
summary_all = sortrows(summary_all, {'Scope','Run','MaxVIF_cond'}, {'ascend','ascend','descend'});

summary_all = summary_all(:, {
    'Scope','Run','Regressor','NRuns', ...
    'MeanVIF_classic','MedianVIF_classic','P95VIF_classic','MaxVIF_classic', ...
    'CountVIFgt5_classic','PctVIFgt5_classic', ...
    'CountVIFgt10_classic','PctVIFgt10_classic', ...
    'MeanVIF_cond','MedianVIF_cond','P95VIF_cond','MaxVIF_cond', ...
    'CountVIFgt5_cond','PctVIFgt5_cond', ...
    'CountVIFgt10_cond','PctVIFgt10_cond' ...
});

writetable(summary_all, summary_csv);
fprintf('Wrote summary (overall + by-run) to: %s\n', summary_csv);