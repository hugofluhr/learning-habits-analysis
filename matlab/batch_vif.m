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
% Group by normalized regressor name
G = findgroups(string(per_run_rows.NormName));

% Helper aggregators (ignore NaNs)
mean_no_nan = @(x) mean(x,'omitnan');
med_no_nan  = @(x) median(x,'omitnan');
max_no_nan  = @(x) max(x,[],'omitnan');
p95_no_nan  = @(x) prctile(x,95);

% Counts over thresholds
over5_classic = @(x) sum(x > thr5, 'omitnan');
over10_classic= @(x) sum(x > thr10,'omitnan');

% Total counts per group
N = splitapply(@(varargin) numel(varargin{1}), per_run_rows.VIF_classic, G);

% Aggregates
mean_classic = splitapply(mean_no_nan, per_run_rows.VIF_classic, G);
med_classic  = splitapply(med_no_nan,  per_run_rows.VIF_classic, G);
max_classic  = splitapply(max_no_nan,  per_run_rows.VIF_classic, G);
p95_classic  = splitapply(p95_no_nan,  per_run_rows.VIF_classic, G);
n5_classic   = splitapply(over5_classic, per_run_rows.VIF_classic, G);
n10_classic  = splitapply(over10_classic, per_run_rows.VIF_classic, G);

mean_cond = splitapply(mean_no_nan, per_run_rows.VIF_conditional, G);
med_cond  = splitapply(med_no_nan,  per_run_rows.VIF_conditional, G);
max_cond  = splitapply(max_no_nan,  per_run_rows.VIF_conditional, G);
p95_cond  = splitapply(p95_no_nan,  per_run_rows.VIF_conditional, G);
n5_cond   = splitapply(over5_classic, per_run_rows.VIF_conditional, G);
n10_cond  = splitapply(over10_classic, per_run_rows.VIF_conditional, G);

reg_names = splitapply(@(s) string(s(1)), string(per_run_rows.NormName), G);

summary = table( ...
    reg_names, N, ...
    mean_classic, med_classic, p95_classic, max_classic, n5_classic, n10_classic, ...
    mean_cond,    med_cond,    p95_cond,    max_cond,    n5_cond,    n10_cond, ...
    'VariableNames', { ...
        'Regressor','NRuns', ...
        'MeanVIF_classic','MedianVIF_classic','P95VIF_classic','MaxVIF_classic','CountVIFgt5_classic','CountVIFgt10_classic', ...
        'MeanVIF_cond','MedianVIF_cond','P95VIF_cond','MaxVIF_cond','CountVIFgt5_cond','CountVIFgt10_cond' ...
    });

% Add percentages
summary.PctVIFgt5_classic  = 100 * summary.CountVIFgt5_classic  ./ summary.NRuns;
summary.PctVIFgt10_classic = 100 * summary.CountVIFgt10_classic ./ summary.NRuns;
summary.PctVIFgt5_cond     = 100 * summary.CountVIFgt5_cond     ./ summary.NRuns;
summary.PctVIFgt10_cond    = 100 * summary.CountVIFgt10_cond    ./ summary.NRuns;

summary = summary(:, {
    'Regressor','NRuns', ...
    'MeanVIF_classic','MedianVIF_classic','P95VIF_classic','MaxVIF_classic', ...
    'CountVIFgt5_classic','PctVIFgt5_classic', ...
    'CountVIFgt10_classic','PctVIFgt10_classic', ...
    'MeanVIF_cond','MedianVIF_cond','P95VIF_cond','MaxVIF_cond', ...
    'CountVIFgt5_cond','PctVIFgt5_cond', ...
    'CountVIFgt10_cond','PctVIFgt10_cond' ...
});

% Sort by worst (conditional) MaxVIF as default
summary = sortrows(summary, 'MaxVIF_cond', 'descend');

% Write summary CSV
writetable(summary, summary_csv);
fprintf('Wrote summary: %s\n', summary_csv);

%% ==== DONE ====
fprintf('Finished. Inspect %s and %s.\n', per_run_csv, summary_csv);