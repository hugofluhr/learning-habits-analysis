clear;

%% ===========================
%% Paths
%% ===========================
spmpath     = '/home/ubuntu/repos/spm12';
analysis_dir= '/home/ubuntu/data/learning-habits/spm_format_noSDC/outputs';
bbt_path    = '/home/ubuntu/data/learning-habits/bbt.csv';
seed_mask = '/home/ubuntu/data/learning-habits/masks/MNI152NLin2009cAsym/putamen_AAL_MNI152NLin2009cAsym.nii';
addpath(spmpath);

%% ===========================
%% Select first-level model to use
%% ===========================
model_name = 'glm2_chosen_all_runs_scrubbed_2025-12-11-11-22';
first_lvl_dir = fullfile(analysis_dir, model_name);
if ~exist(first_lvl_dir, 'dir')
    error('No directories found matching %s', fullfile(analysis_dir, model_name));
end
disp(['Using first-level output_dir: ' first_lvl_dir]);

%% ===========================
%% Seed mask + VOI settings
%% ===========================
voi_base_name = 'putamen';                   % VOI name prefix in output
%contrast_name = 'Effects of interest';       % must match your F-contrast name

mask_threshold = 0.5;  % if binary mask, 0.5 is typical

%% ===========================
%% Logging
%% ===========================
log_path = fullfile(first_lvl_dir, 'extract_voi_seed_log.txt');
diary(log_path);
diary on;

%% ===========================
%% Load behavioral data + subjects
%% ===========================
bbt = readtable(bbt_path);
subjects = unique(bbt.sub_id);

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

for s = 1:length(subjects)
    sub_id = subjects{s};

    if strcmp(sub_id,'sub-04') || strcmp(sub_id,'sub-45')
        continue;
    end

    disp(['Extracting VOI for subject: ' sub_id]);

    sub_output_dir = fullfile(first_lvl_dir, sub_id);
    SPM_path = fullfile(sub_output_dir, 'SPM.mat');

    if ~exist(SPM_path, 'file')
        warning(['Missing SPM.mat for ' sub_id]);
        continue;
    end

    load(SPM_path);

    %% ===========================
    %% Find Effects-of-interest contrast index by name
    %% ===========================
    % con_idx = [];
    % if isfield(SPM, 'xCon') && ~isempty(SPM.xCon)
    %     for c = 1:numel(SPM.xCon)
    %         if strcmp(SPM.xCon(c).name, contrast_name)
    %             con_idx = c;
    %             break;
    %         end
    %     end
    % end

    % if isempty(con_idx)
    %     warning(['No contrast named "' contrast_name '" found for ' sub_id '. Skipping.']);
    %     continue;
    % end

    %% ===========================
    %% Extract VOI per session
    %% ===========================
    nSess = numel(SPM.Sess);
    for sess = 1:nSess

        voi_name = sprintf('%s_sess%d', voi_base_name, sess);

        % If you re-run the script, SPM will create new VOI_* files with incremented suffix.
        % If you want to skip existing, you can add a check here.

        matlabbatch = [];
        matlabbatch{1}.spm.util.voi.spmmat = {SPM_path};
        matlabbatch{1}.spm.util.voi.adjust = 0;  % no adjustment for now
        matlabbatch{1}.spm.util.voi.session = sess;
        matlabbatch{1}.spm.util.voi.name = voi_name;

        matlabbatch{1}.spm.util.voi.roi{1}.mask.image = {seed_mask};
        matlabbatch{1}.spm.util.voi.roi{1}.mask.threshold = mask_threshold;

        matlabbatch{1}.spm.util.voi.expression = 'i1';

        try
            spm_jobman('run', matlabbatch);
            disp(['  OK: ' voi_name]);
        catch ME
            warning(['  FAILED: ' voi_name ' (' sub_id ')']);
            disp(getReport(ME, 'extended'));
        end
    end
end

diary off;
disp('Done extracting VOIs.');