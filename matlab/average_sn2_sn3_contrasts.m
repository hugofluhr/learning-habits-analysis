% average_sn2_sn3_contrasts.m
%
% Averages matching contrast images from session-02 and session-03 into a
% new session-02-03 directory.
%
% Expected structure:
%   session-02/
%     contrast-XX_<name>/
%       sub-YY_desc-<name>_con.nii
%   session-03/   (same layout)
%
% Output mirrors the same layout under session-02-03/.
%
% Usage: set root_dir to the model export root, then run.

%% ---- configuration -------------------------------------------------------

root_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/spm_outputs_noSDC/session_split/glm2_chosen_all_runs_scrubbed_2025-12-11-11-22';

sn2_dir  = fullfile(root_dir, 'session-02');
sn3_dir  = fullfile(root_dir, 'session-03');
out_dir  = fullfile(root_dir, 'session-02-03');

%% ---- setup ---------------------------------------------------------------

if ~exist(out_dir, 'dir')
    mkdir(out_dir);
    fprintf('Created output directory: %s\n', out_dir);
end

% list contrast subdirectories in session-02
contrast_dirs = dir(sn2_dir);
contrast_dirs = contrast_dirs([contrast_dirs.isdir] & ~startsWith({contrast_dirs.name}, '.'));

if isempty(contrast_dirs)
    error('No contrast subdirectories found in %s', sn2_dir);
end

fprintf('Found %d contrast subdirectories in session-02\n', numel(contrast_dirs));

%% ---- loop over contrasts then subjects -----------------------------------

n_missing  = 0;
n_averaged = 0;

for c = 1:numel(contrast_dirs)
    con_name = contrast_dirs(c).name;

    sn2_con_dir = fullfile(sn2_dir, con_name);
    sn3_con_dir = fullfile(sn3_dir, con_name);
    out_con_dir = fullfile(out_dir, con_name);

    if ~exist(sn3_con_dir, 'dir')
        warning('No matching session-03 directory for %s — skipping.', con_name);
        n_missing = n_missing + 1;
        continue
    end

    if ~exist(out_con_dir, 'dir')
        mkdir(out_con_dir);
    end

    % list subject .nii files within this contrast directory
    subj_files = dir(fullfile(sn2_con_dir, '*.nii'));

    for s = 1:numel(subj_files)
        fname = subj_files(s).name;

        sn2_path = fullfile(sn2_con_dir, fname);
        sn3_path = fullfile(sn3_con_dir, fname);
        out_path = fullfile(out_con_dir, fname);

        if ~exist(sn3_path, 'file')
            warning('No matching session-03 file for %s / %s — skipping.', con_name, fname);
            n_missing = n_missing + 1;
            continue
        end

        % average using spm_imcalc: (i1 + i2) / 2
        spm_imcalc({sn2_path; sn3_path}, out_path, '(i1+i2)/2');
        n_averaged = n_averaged + 1;
    end

    fprintf('  %s: %d subjects averaged\n', con_name, numel(subj_files));
end

%% ---- summary -------------------------------------------------------------

fprintf('\nDone.\n');
fprintf('  Averaged : %d\n', n_averaged);
fprintf('  Missing  : %d\n', n_missing);
fprintf('  Output   : %s\n', out_dir);
