% Pmod check for the concatenated-design first level (one subject).
%
% Injected variables (cluster defaults below):
%   concat_dir - subject folder of the concat model (contains SPM.mat)
%   sub_id     - subject id (default 'sub-01')
%   bbt_path   - bbt csv
%
% Prints:
%   1. per-run and overall mean/std of the z-scored pmod columns in the bbt
%      (is the z-scoring per run or across runs?) and the share of each column's
%      variance that lies between runs
%   2. mean/std of the pmod and parent-stim design columns per run
%   3. correlation of each pmod column with its parent-stim column, overall and per run
%
% VIFs are in compare_concat_vifs.py.
if ~exist('concat_dir', 'var') || isempty(concat_dir)
    concat_dir = '/home/hfluhr/data/learninghabits/spm_format/outputs/glm2_all_runs_concat_scrubbed_2026-09-24-11-16/sub-01';
end
if ~exist('sub_id', 'var') || isempty(sub_id)
    sub_id = 'sub-01';
end
if ~exist('bbt_path', 'var') || isempty(bbt_path)
    bbt_path = '/home/hfluhr/data/learninghabits/bbt_062026_mf_cols.csv';
end
addpath('/home/hfluhr/repos/spm12');

load(fullfile(concat_dir, 'SPM.mat'));
n = SPM.xX.name; X = SPM.xX.X; e = cumsum(SPM.nscan); s0 = [0 e(1:end-1)];
bbt = readtable(bbt_path); bl = {'learning1', 'learning2', 'test'};
vars = {'first_stim_value_rl_zscore', 'first_stim_value_ck_zscore', 'second_stim_value_rl_zscore', 'second_stim_value_ck_zscore'};

fprintf('== 1. bbt pmod columns, %s: mean/std per run, overall, and between-run variance share ==\n', sub_id);
for v = 1:numel(vars)
    allv = []; grp = [];
    for r = 1:3
        b = bbt(strcmp(bbt.sub_id, sub_id) & strcmp(bbt.block, bl{r}), :);
        x = b.(vars{v}); allv = [allv; x]; grp = [grp; r * ones(numel(x), 1)];
        fprintf('%-28s run%d: n=%d mean=%+.3f std=%.3f nNaN=%d\n', vars{v}, r, numel(x), mean(x, 'omitnan'), std(x, 'omitnan'), sum(isnan(x)));
    end
    gm = mean(allv, 'omitnan'); between = 0;
    for r = 1:3; xr = allv(grp == r); between = between + numel(xr) * (mean(xr, 'omitnan') - gm)^2; end
    fprintf('%-28s ALL : n=%d mean=%+.3f std=%.3f  between-run variance share=%.3f\n', vars{v}, numel(allv), gm, std(allv, 'omitnan'), between / sum((allv - gm).^2, 'omitnan'));
end

fprintf('\n== 2. design columns for the 4 pmods and their parent stims ==\n');
cols = find(contains(n, 'first_stim') | contains(n, 'second_stim'));
for c = cols
    fprintf('col %d %-32s mean=%+.4f std=%.4f  std per run: %s\n', c, n{c}, mean(X(:, c)), std(X(:, c)), mat2str(arrayfun(@(r) std(X(s0(r)+1:e(r), c)), 1:3), 3));
end

fprintf('\n== 3. correlation of each pmod col with its parent stim col (all rows / per run) ==\n');
pairs = {'first_stimxQval', 'first_stim'; 'first_stimxHval', 'first_stim'; 'second_stimxQval', 'second_stim'; 'second_stimxHval', 'second_stim'};
for p = 1:4
    a = find(contains(n, pairs{p, 1})); b = find(contains(n, pairs{p, 2}) & ~contains(n, 'xQval') & ~contains(n, 'xHval'));
    rr = arrayfun(@(r) corr(X(s0(r)+1:e(r), a), X(s0(r)+1:e(r), b)), 1:3);
    fprintf('%-18s vs %-12s: all=%+.3f  per run=%s\n', pairs{p, 1}, pairs{p, 2}, corr(X(:, a), X(:, b)), mat2str(rr, 3));
end
