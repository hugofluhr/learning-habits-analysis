clear;

% Which confounds to use
confound_pattern = '_.*_motion_with_dummies.txt$';

% Paths
spmpath = '/home/ubuntu/repos/spm12';
data_dir = '/home/ubuntu/data/learning-habits/spm_format_noSDC';
analysis_dir = '/home/ubuntu/data/learning-habits/spm_format_noSDC';
bbt_path = '/home/ubuntu/data/learning-habits/bbt.csv';
addpath(spmpath);

current_date = char(datetime('now', 'Format', 'yyyy-MM-dd-hh-mm'));
output_dir = fullfile(analysis_dir, 'outputs', ['glm_sc2_faces_' current_date]);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Start logging
log_path = fullfile(output_dir, 'matlab_log.txt');
diary(log_path);
diary on;

% Load behavioral data
bbt = readtable(bbt_path);
bbt.first_stim_face = double(strcmp(bbt.first_stim_cat, 'face'));
bbt.second_stim_face = double(strcmp(bbt.second_stim_cat, 'face'));
subjects = unique(bbt.sub_id);
block_names = {'learning1', 'learning2', 'test'};

% Parameters
TR = 2.33384; % Repetition time
high_pass_cutoff = 128; % High-pass filter in seconds

% set up contrasts - not including non-response feedback because so few trials per subject
% points_feedback excluded from contrasts as it is absent in test run
connames = {
    'first_stim', ...
    'second_stim', ...
    'response', 'purple_frame'
    };

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

% Loop through subjects
for s = 1:length(subjects)
    sub_id = subjects{s};
    disp(['Processing subject: ', sub_id]);
    func_dir = fullfile(data_dir, sub_id, 'func');
    
    % Identify unique runs based on BOLD file names
    bold_files = spm_select('FPList', func_dir, '^sub-.*_desc-preproc_bold.nii$');
    run_ids = unique(regexp(cellstr(bold_files), 'run-\d+', 'match', 'once'));
    
    %% ===========================
    %% FULL MODEL (run01+run02)  ->  outputs/<sub>/
    %% ===========================
    sub_output_dir = fullfile(output_dir, sub_id);
    if ~exist(sub_output_dir, 'dir'); mkdir(sub_output_dir); end

    % Initialize batch
    clear matlabbatch
    matlabbatch{1}.spm.stats.fmri_spec.dir = {sub_output_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
    % TO DO: check if these are correct, these are default values
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
        
    for r = 1:3 % all runs (learning1, learning2, test)
        run_id = run_ids{r};
        % Select BOLD files for the current run
        currBOLD = spm_select('FPList', func_dir, ['^smoothed_.*' run_id '_.*_bold.nii$']);
        brain_mask = spm_select('FPList', func_dir, ['^sub-.*' run_id '_.*_desc-brain_mask.nii$']);
        confounds_file = spm_select('FPList', func_dir, ['^sub-.*' run_id confound_pattern]);
        
        % Sanity check for missing files
        if isempty(currBOLD)
            warning(['No BOLD files found for ', run_id, ' in subject ', sub_id]);
            continue;
        end
        if isempty(brain_mask)
            warning(['Brain mask not found for ', run_id, ' in subject ', sub_id]);
            continue;
        end
        if isempty(confounds_file)
            warning(['Confounds not found for ', run_id, ' in subject ', sub_id]);
            continue;
        end
        
        % Get block data and filter trials
        block_data = bbt(strcmp(bbt.sub_id, sub_id) & strcmp(bbt.block, block_names{r}), :);
        % Model non response trials separately
        is_resp = ~isnan(block_data.action);
        block_resp = block_data(is_resp, :);
        block_nr   = block_data(~is_resp, :);

        % General stuff
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).scans = cellstr(currBOLD);
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).multi_reg = {confounds_file};
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).hpf = high_pass_cutoff;

        % First stimulus - All trials
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).name = 'first_stim';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).onset = block_data.t_first_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).duration = block_data.t_second_stim - block_data.t_first_stim;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(1).orth = 0;

        % Second stimulus - All trials
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).name = 'second_stim';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).onset = block_data.t_second_stim;
        % handle duration
        duration = block_data.t_action - block_data.t_second_stim;
        duration(duration < 0) = 1; % for non-response trials
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).duration = duration;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(2).orth = 0;

        % Response - Resp trials
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).name = 'response';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).onset = block_resp.t_action;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).duration = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(3).orth = 0;

        % Purple frame - Resp trials
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).name = 'purple_frame';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).onset = block_resp.t_purple_frame;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).duration = block_resp.t_points_feedback - block_resp.t_purple_frame;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(4).orth = 0;

        % Points feedback - Resp trials - different for learning and test
        if r == 3
            % Test run - no points feedback -> set as empty
            k = 5; % next condition index
        else
            % Learning runs
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).name = 'points_feedback';
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).onset = block_resp.t_points_feedback;
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).duration = block_resp.t_iti_onset - block_resp.t_points_feedback;
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).tmod = 0;
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).pmod = struct('name', {}, 'param', {}, 'poly', {});
            matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(5).orth = 0;
            k = 6; % next condition index
        end

        % Feedback - NoResp trials
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k).name = 'nresp_screen';
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k).onset = block_nr.t_second_stim + 1; % No time stamp for non-response trials, so use second stimulus + 1s
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k).duration = block_nr.t_iti_onset - block_nr.t_second_stim - 1;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k).orth = 0;

        % Seeing face - from first appearance of a face to ITI onset
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k+1).name = 'face';
        % Face present if either stimulus is a face
        is_face = (block_data.first_stim_face == 1) | (block_data.second_stim_face == 1);
        % Compute onset per trial, then subset
        face_onset_all = block_data.t_first_stim;
        % If face appears only at second stim, onset should be second stim
        second_only_face = (block_data.first_stim_face == 0) & (block_data.second_stim_face == 1);
        face_onset_all(second_only_face) = block_data.t_second_stim(second_only_face);
        face_onset = face_onset_all(is_face);
        % Duration from face onset to ITI onset
        face_duration = block_data.t_iti_onset(is_face) - face_onset;
        face_duration(face_duration < 0) = 0; % safety
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k+1).onset = face_onset;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k+1).duration = face_duration;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k+1).tmod = 0;
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k+1).pmod = struct('name', {}, 'param', {}, 'poly', {});
        matlabbatch{1}.spm.stats.fmri_spec.sess(r).cond(k+1).orth = 0;
    end

    % Other specifications, from Jae-Chang's script
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mask = cellstr(brain_mask);
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
    
    %% Model estimation
    matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep('fMRI model specification: SPM.mat File', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','spmmat'));
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    
    save(fullfile(sub_output_dir, 'batch_spec_est.mat'), 'matlabbatch');
    spm_jobman('run', matlabbatch);
    
    % Contrasts (combined) 
    SPM_path = fullfile(sub_output_dir, 'SPM.mat');
    load(SPM_path);

    matlabbatch_con = [];
    matlabbatch_con{1}.spm.stats.con.spmmat = {fullfile(sub_output_dir, 'SPM.mat')};
    colnames = SPM.xX.name;
    nCols    = numel(colnames);
    nSess    = numel(SPM.Sess);

    for cc = 1:length(connames)
        cname = connames{cc};
        w_all = zeros(1, nCols);   % full-length contrast vector
        
        for s = 1:nSess
            cols_s  = SPM.Sess(s).col;        % global column indices for this session
            names_s = colnames(cols_s);       % names for this session

            % Just match the condition name (main or pmod)
            idx_local = find(contains(names_s, cname), 1);  

            if isempty(idx_local)
                % regressor absent in this session - always create a warning for safety
                warning('Regressor "%s" not found in session %d', cname, s);
                continue
            end

            col_global = cols_s(idx_local);
            w_all(col_global) = 1;
        end
        
        if all(w_all == 0)
            warning('No columns found for contrast "%s"', cname);
            continue
        end

        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.name    = cname;
        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.weights = w_all;
        matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.sessrep = 'none';
    end

    matlabbatch_con{1}.spm.stats.con.delete = 0;
    save(fullfile(sub_output_dir, 'batch_contrasts.mat'), 'matlabbatch_con');
    spm_jobman('run', matlabbatch_con);

    disp(['Model for ', sub_id, ' complete.']);
    
end

diary off;