clear;

%% ===========================
%% Paths
%% ===========================
% spmpath      = '/home/ubuntu/repos/spm12';
% pppi_path    = '/home/ubuntu/repos/PPPIv13.1';
% analysis_dir = '/home/ubuntu/data/learning-habits/spm_format_noSDC/outputs';
% bbt_path     = '/home/ubuntu/data/learning-habits/bbt.csv';
spmpath      = '/Users/hugofluhr/code/spm12';
pppi_path    = '/Users/hugofluhr/phd_local/repositories/PPPI';
analysis_dir = '/home/ubuntu/data/learning-habits/spm_format_noSDC/outputs';
bbt_path     = '/home/ubuntu/data/learning-habits/bbt.csv';

addpath(spmpath);
addpath(genpath(pppi_path));

%% ===========================
%% FIRST-LEVEL model to use
%% ===========================
% This should be the multi-session GLM whose timing/events you want to use for gPPI.
% (Typically your chosen-stim model; if you?ve got a ?chosen model without Qval? use that.)
model_name = 'glm2_chosen_all_runs_scrubbed_2025-12-11-11-22';
first_lvl_dir = fullfile(analysis_dir, model_name);
if ~exist(first_lvl_dir, 'dir')
    error('First-level model folder not found: %s', first_lvl_dir);
end
disp(['Using base first-level dir: ' first_lvl_dir]);

%% ===========================
%% Create output directory
%% ===========================
ts = datestr(now,'yyyy-mm-dd-HH-MM-SS');
gppi_root = fullfile(analysis_dir, ['PPI/gppi_putamen_Hvalchosen_deconv_' ts]);
if ~exist(gppi_root,'dir'), mkdir(gppi_root); end
disp(['gPPI output root: ' gppi_root]);


%% ===========================
%% Logging
%% ===========================
log_path = fullfile(gppi_root, 'gppi_log.txt');
diary(log_path);
diary on;

%% ===========================
%% Subjects and behavioral data
%% ===========================
bbt = readtable(bbt_path);
subjects = unique(bbt.sub_id);

spm('Defaults','fMRI');
spm_jobman('initcfg');

%% ===========================
%% Seed region
%% ===========================
seed_region_name = 'putamen';
seed_region_mask = '/mnt/data/learning-habits/masks/MNI152NLin2009cAsym/putamen_AAL_MNI152NLin2009cAsym.nii';
if ~exist(seed_region_mask,'file')
    error('Seed region mask not found: %s', seed_region_mask);
end
disp(['Using seed region mask: ' seed_region_mask]);

%% ===========================
%% gPPI settings (common across subjects)
%% ===========================
P = struct();
P.method = 'cond'; % 'trad' for traditional SPM PPI, 'cond' for gPPI
P.analysis = 'psy';
P.extract = 'eig';
P.Region = seed_region_name;
P.VOI = seed_region_mask;
P.FLmask = 1; % use the first-level mask to constrain VOI extraction
P.equalroi = 0; % allow seed to be different size across subjects
% P.contrast = 0; % no adjustment for now. commented to leave default
P.Estimate = 1; % estimate the gPPI model immediately
P.CompContrasts = 1; % compute contrasts immediately
% Defining the events (called Tasks)
P.Tasks = { '0',... % PPPI convention
            'first_stim', ...
            'second_stim', ...
            'second_stimxHval_chosen^1', ...
            'response', ...
            'purple_frame', ...
            'points_feedback' ...
            };

%% ===========================
%% Defining contrasts
%% ===========================
% Define contrasts in a loop
for j = 1:(numel(P.Tasks)-1) % skip the '0' condition
    P.Contrasts(j).name = P.Tasks{j+1};
    P.Contrasts(j).left = {P.Tasks{j+1}};
    P.Contrasts(j).right = {'none'};
    P.Contrasts(j).STAT = 'T';
    P.Contrasts(j).Weighted = 0;
    P.Contrasts(j).MinEvents = 1;
end

%% ===========================
%% Run gPPI for each subject
%% ===========================
for i = 1:numel(subjects)
    sub_id = subjects{i};
    disp('==================================================');
    disp(['Processing subject: ' sub_id]);
    
    if strcmp(sub_id,'sub-04') || strcmp(sub_id,'sub-45')
        disp('Skipping subject (known issues): ' + string(sub_id));
        continue;
    end

    % Base-design SPM.mat (multi-session)
    sub_base_dir = fullfile(first_lvl_dir, sub_id);
    spm_mat = fullfile(sub_base_dir, 'SPM.mat');
    if ~exist(spm_mat, 'file')
        warning('SPM.mat not found for subject %s: %s. Skipping.', sub_id, spm_mat);
        continue;
    end

    % Subject output directory for gPPI results
    sub_outdir = fullfile(gppi_root, sub_id);
    if ~exist(sub_outdir,'dir')
        mkdir(sub_outdir);
    end
    % To avoid saving files elsewhere
    cd(sub_outdir);

    % ---------------------------------------------------------------------
    % To keep your gPPI outputs separated, we copy SPM.mat into the new
    % subject gPPI folder and point PPPI there.
    % ---------------------------------------------------------------------
    copyfile(spm_mat, fullfile(sub_outdir, 'SPM.mat'));

    % Also copy first-level files PPPI may need
    aux_files = {'mask.nii','ResMS.nii'};
    for k = 1:numel(aux_files)
        f = fullfile(sub_base_dir, aux_files{k});
        if exist(f,'file')
            copyfile(f, fullfile(sub_outdir, aux_files{k}));
        end
    end

    % Set subject-specific fields in P
    Psub = P; % start with the common settings
    Psub.subject = sub_id;
    Psub.directory = sub_outdir; % to ensure the first-lvl directory is untouched
    Psub.outdir = sub_outdir; % maybe not needed

    % Save P structure for this subject
    save(fullfile(sub_outdir, 'P.mat'), 'Psub');

    % Run PPPI
    try
        PPPI(Psub);
        disp(['gPPI completed for subject: ' sub_id]);
    catch ME
        warning('Error running gPPI for subject %s: %s', sub_id, ME.message);
    end
end

diary off;