function sfnr_plot(par_workers)
% Parallel plotting of SFNR maps saved by sfnr_calc() using consistent color limits
%
% Usage:
%   sfnr_plot            % auto-detect or reuse existing limits
%   sfnr_plot(8)         % force parpool with 8 workers
%
% Behavior:
% - Scans plot_save_dir/*/run-*/sfnr.nii
% - Uses FIXED slices: [3,5,7,9,11,13,17,21,25,29,33,37]
% - Pass 1 (parallel): pooled sampling to get global color limits (1st–99th pct)
%   * If plot_save_dir/clim.txt exists, reuse its limits and skip sampling.
% - Pass 2 (parallel): renders 3x4 slice montages per run with fixed limits.
%
% Requirements: Parallel Computing Toolbox, SPM on path

%% ---- CONFIG ----
addpath('~/code/spm25/');  % SPM path (adjust if needed)
plot_save_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/sfnr';

slices = [3,5,7,9,11,13,17,21,25,29,33,37]; % fixed slices
n_rows = 3;
n_cols = 4;

low_pct = 1;              % lower percentile for color limits
high_pct = 99;            % upper percentile for color limits
samples_per_img = 10000;  % voxel samples per image to pool
rng_seed = 0;             % make sampling reproducible across runs

clim_txt = fullfile(plot_save_dir, 'clim.txt');

%% ---- PARPOOL SETUP ----
if nargin < 1 || isempty(par_workers)
    % If a pool exists, keep it; otherwise create default
    pool = gcp('nocreate');
    if isempty(pool)
        parpool; %#ok<*NOPRT>
    end
else
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= par_workers
        if ~isempty(pool), delete(pool); end
        parpool(par_workers);
    end
end

%% ---- FIND ALL SFNR FILES ----
sfnr_files = dir(fullfile(plot_save_dir, '*', 'run-*', 'sfnr.nii'));
if isempty(sfnr_files)
    error('No sfnr.nii files found under %s', plot_save_dir);
end
N = numel(sfnr_files);
fprintf('Found %d SFNR files.\n', N);

%% ---- PASS 1: GLOBAL COLOR LIMITS (reuse or compute in parallel) ----
reuse_clim = false;
if exist(clim_txt, 'file') == 2
    try
        T = readtable(clim_txt, 'FileType', 'text', 'Delimiter', '\t');
        if all(ismember({'clow','chigh'}, T.Properties.VariableNames))
            clow = T.clow(1);
            chigh = T.chigh(1);
            reuse_clim = true;
            fprintf('Reusing color limits from clim.txt: [%.3f, %.3f]\n', clow, chigh);
        end
    catch
        % fall through to recompute
    end
end

if ~reuse_clim
    fprintf('Sampling voxels in parallel to set global color limits (p%d–p%d)...\n', low_pct, high_pct);

    samples = cell(N,1);
    parfor k = 1:N
        f = fullfile(sfnr_files(k).folder, sfnr_files(k).name);
        local = [];
        try
            V = spm_vol(f);
            Y = spm_read_vols(V);
            y = Y(:);
            y = y(isfinite(y) & y > 0);
            if ~isempty(y)
                % Reproducible sampling per file
                s = RandStream('Threefry','Seed',rng_seed + k);
                if numel(y) > samples_per_img
                    idx = randperm(s, numel(y), samples_per_img);
                    local = y(idx);
                else
                    local = y;
                end
            end
        catch ME
            warning('Sampling failed for %s (%s).', f, ME.message);
        end
        samples{k} = local;
    end

    pooled = vertcat(samples{:});
    if isempty(pooled)
        error('No valid SFNR voxels found to compute color limits.');
    end

    clow  = prctile(pooled, low_pct);
    chigh = prctile(pooled, high_pct);
    clow = max(0, clow);
    if chigh <= clow
        chigh = clow + 1;
    end
    fprintf('Global color limits: [%.3f, %.3f]\n', clow, chigh);

    % Save limits for provenance
    fid = fopen(clim_txt, 'w');
    fprintf(fid, 'clow\tchigh\tlow_pct\thigh_pct\tsamples_per_img\n');
    fprintf(fid, '%.6f\t%.6f\t%d\t%d\t%d\n', clow, chigh, low_pct, high_pct, samples_per_img);
    fclose(fid);
end

%% ---- PASS 2: PARALLEL RENDERING ----
fprintf('Rendering %d montages in parallel...\n', N);

parfor k = 1:N
    f = fullfile(sfnr_files(k).folder, sfnr_files(k).name);

    % Infer subject and run from folder names
    [run_dir, ~] = fileparts(f);
    [subj_dir, run_name] = fileparts(run_dir);
    [~, subj_name] = fileparts(subj_dir);

    % Extract run number for filename if possible
    run_no = NaN;
    tok = regexp(run_name, 'run-(\d+)', 'tokens', 'once');
    if ~isempty(tok), run_no = str2double(tok{1}); end

    try
        V = spm_vol(f);
        Y = spm_read_vols(V);
        nz = size(Y,3);
        valid_slices = slices(slices <= nz);

        % Use a figure per worker; invisible to avoid GUI overhead
        h = figure('Visible', 'off', 'Color', 'w', 'Units', 'pixels', 'Position', [100 100 1400 900]);
        t = tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');

        for i = 1:numel(valid_slices)
            nexttile;
            slice_i = valid_slices(i);
            img = Y(:,:, slice_i);
            imagesc(rot90(img, -1), [clow chigh]);
            axis image off;
            title(sprintf('slice %d', slice_i), 'Interpreter', 'none', 'FontSize', 10);
        end
        colormap(t.Parent, 'jet');
        cb = colorbar('Location','eastoutside'); cb.Label.String = 'SFNR';
        title(t, sprintf('SFNR  |  sub %s  |  %s', subj_name, run_name), 'FontSize', 12, 'FontWeight', 'bold');

        % Output filename
        if ~isnan(run_no)
            out_png = fullfile(run_dir, sprintf('SFNR_check_S%s_R%i.png', subj_name, run_no));
        else
            out_png = fullfile(run_dir, sprintf('SFNR_check_S%s_%s.png', subj_name, run_name));
        end

        exportgraphics(h, out_png, 'Resolution', 800);
        close(h);
    catch ME
        warning('Failed to plot %s (%s).', f, ME.message);
        % Ensure figure is closed on error
        try, if exist('h','var') && isvalid(h), close(h); end, catch, end
    end
end

fprintf('Done.\n');
end