function run_1L_trialwise_LSS(base_path, spm_path, sub, TR, trial_id, invert_flag, task_name, fmriprep_subdir)
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Wrapper for SPM first-level analysis: spm specification & estimation.
    %
    %    run_1L_trialwise_LSS(base_path, spm_path, sub, TR, trial_id, invert_flag, task_name, fmriprep_subdir)
    %
    % Args:
    %   base_path       [str] Path to data, containing /bids /derivatives dirs.
    %   spm_path        [str] Path to /spm (to read tpm/mask_ICV.nii)
    %   sub             [chr] must match dir name ('sub-001')
    %   TR              [dbl] Repetition time in seconds (all full-sample subjects: 1.42)
    %   trial_id        [dbl] Range: [1-152]
    %   invert_flag     [bol] Opt: boolean to use inverted compound model (default: false)
    %   task_name       [str] Opt: BIDS task label (default: 'causal'; pilot: 'cond')
    %   fmriprep_subdir [str] Opt: path under derivatives/ to fmriprep (default: 'no_sdc/fmriprep'; pilot: 'fmriprep25/no_fmap_correction')
    %
    % NOTE on spm_select: it uses regex matching on filenames. The unsmoothed BOLD
    % pattern is anchored with '^' so that pre-smoothed 's4_sub-*' files in the same
    % directory are not accidentally included, which would double-count scans.
    %
    % MODEL: runs a trialwise LSS FLM modeling cue and outcome of the
    %  specified trial, while adding all other trial per condition as
    %  'confound' main regressors.
    % SPECS: uses classical HRF.
    %
    % 24.04.2026
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    if nargin < 6 || isempty(invert_flag),     invert_flag     = false;                   end
    if nargin < 7 || isempty(task_name),       task_name       = 'causal';                end
    if nargin < 8 || isempty(fmriprep_subdir), fmriprep_subdir = 'no_sdc/fmriprep';      end

    analysis_name = 'trialwise_LSS_cue_outcome_cHRF';                      % 'trialwise_LSS_cue_outcome_cdHRF' 'trialwise_LSS_cue_outcome_firHRF'
    
    out_dir = fullfile(base_path, 'derivatives', 'spm25', analysis_name, '1L', sub, sprintf('lss_trial%03d', trial_id));

    % --- check whether trial already completed ---
    % ResMS.nii is written by SPM after ALL beta images, making it a reliable
    % completion marker. Accept .nii and .nii.gz (latter after post-run zipping).
    resms    = fullfile(out_dir, 'ResMS.nii');
    resms_gz = fullfile(out_dir, 'ResMS.nii.gz');
    if exist(out_dir, 'dir') && (exist(resms, 'file') || exist(resms_gz, 'file'))
        fprintf('Already estimated for %s trial %03d — skipping.\n', sub, trial_id);
        return;
    elseif exist(out_dir, 'dir')
        % Directory exists but no ResMS — incomplete/crashed run: wipe and restart.
        fprintf('Incomplete run for %s trial %03d — cleaning up and re-running.\n', sub, trial_id);
        rmdir(out_dir, 's');
        mkdir(out_dir);
    else
        mkdir(out_dir)
    end

    % --- initialise SPM ---
    addpath(spm_path)
    spm('defaults','fmri');
    spm_jobman('initcfg');

    % --- specify First-level GLM ---
    % Put the run containing the TOI first so it lands in session 1
    % and its betas are always beta_0001 (cue) and beta_0002 (outcome).
    onsetFolder_tmp = fullfile(base_path, 'bids');
    [~, ~, o_run1, ~] = get_trial_info(sub, 1, onsetFolder_tmp, 'cue', trial_id, invert_flag);
    if ~isempty(o_run1)
        RUNS = [1, 2];
    else
        RUNS = [2, 1];
    end
    create_SPM_design_matrix(base_path, spm_path, out_dir, sub, RUNS, TR, trial_id, invert_flag, task_name, fmriprep_subdir);
    fprintf('SPM design matrix of %s has been created.\n', sub);

    % --- estimate First-level GLM ---
    estimate_from_SPM_design_matrix(out_dir);
    fprintf('GLM parameters of %s have been estimated.\n', sub);

    % --- zip SPM output NIfTIs to save disk space ---
    zip_spm_outputs(out_dir);
end

function create_SPM_design_matrix(path, spm_path, out_dir, sub, valid_runs, TR, trial_id, invert_flag, task_name, fmriprep_subdir)
    % Specifies the SPM design matrix for a single trial.

    % path to imaging data
    funcPath = fullfile(path, 'derivatives', fmriprep_subdir, sub, 'func');

    matlabbatch{1}.spm.stats.fmri_spec.dir = {out_dir}; 
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = TR;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;

    for counter = 1:numel(valid_runs)
        run = valid_runs(counter);
        sess = struct();
        onsetFolder = fullfile(path, 'bids');

        % unzip preprocessed data if necessary
        nii_gz_file = fullfile(funcPath, [sub '_task-' task_name '_run-0' num2str(run) '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz']); 
        nii_file    = fullfile(funcPath, [sub '_task-' task_name '_run-0' num2str(run) '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii']);    
        if exist(nii_gz_file, 'file') && ~exist(nii_file, 'file')
            fprintf('Unzipping %s...\n', nii_gz_file);
            gunzip(nii_gz_file);
        end
        
        % unsmoothed BOLD: '^' anchors the regex so spm_select does not
        % accidentally match pre-smoothed 's4_sub-*' files in the same dir
        scans = cellstr(spm_select('ExtFPList', funcPath, ...
            ['^' sub '_task-' task_name '_run-0' num2str(run) '_space-MNI152NLin2009cAsym_desc-preproc_bold\.nii'], Inf));
        sess.scans = scans; 

        % load fMRIPrep confounds
        confounds_file = fullfile(funcPath, [sub '_task-' task_name '_run-0' num2str(run) '_desc-confounds_timeseries.tsv']);
        rp = readtable(confounds_file, 'FileType', 'text', 'Delimiter', '\t');

        % replace NaNs (from 'n/a' in fMRIPrep) with 0 across all numeric columns
        for c = 1:width(rp)
            if isnumeric(rp.(c))
                rp.(c)(isnan(rp.(c))) = 0;
            end
        end

        % --- specify conditions ---
        c=1;
        % Condition 1: CUE - trial of interest 
        [cond_name, cue_name, o, d] = get_trial_info(sub, run, onsetFolder, 'cue', trial_id, invert_flag); 
        if ~isempty(o) && ~isempty(d)
            sess.cond(c).name = sprintf('cue-trial%03d_cond-%s_stim-%s', trial_id, cond_name, cue_name);
            sess.cond(c).onset = o;
            sess.cond(c).duration = d;
            sess.cond(c).tmod = 0;
            sess.cond(c).pmod = struct('name', {}, 'param', {}, 'poly', {});
            sess.cond(c).orth = 0;
            c=c+1;
        end

        % Condition 2: OUTCOME - trial of interest
        [out_name, ~, o, d] = get_trial_info(sub, run, onsetFolder, 'outcome', trial_id, invert_flag);
        if ~isempty(o) && ~isempty(d)
            sess.cond(c).name = sprintf('outcome-trial%03d_%s', trial_id, out_name);
            sess.cond(c).onset = o;
            sess.cond(c).duration = d;
            sess.cond(c).tmod = 0;
            sess.cond(c).pmod = struct('name', {}, 'param', {}, 'poly', {});
            sess.cond(c).orth = 0;
            c=c+1;
        end

        % Condition 2: CUES - same stimuli as TOI
        [o, d] = get_onsets_except(sub, run, onsetFolder, cond_name, trial_id, invert_flag); 
        sess.cond(c).name = sprintf('cues_cond-%s', cond_name);
        sess.cond(c).onset = o;
        sess.cond(c).duration = d;
        sess.cond(c).tmod = 0;
        sess.cond(c).pmod = struct('name', {}, 'param', {}, 'poly', {});
        sess.cond(c).orth = 0;
        c=c+1;

        % Condition 3: CUES - other stimuli as TOI 
        cue_cond_names = {'single-high', 'single-low', 'single-medium', 'compound-high', 'compound-low'};
        mask = ~strcmp(cue_cond_names, cond_name);
        other_conds_cue = cue_cond_names(mask);
        for i=1:length(other_conds_cue)
            c_name = other_conds_cue{i};
            [o, d] = get_onsets_except(sub, run, onsetFolder, c_name, [], invert_flag); 
            sess.cond(c).name = sprintf('cues_cond-%s', c_name);
            sess.cond(c).onset = o;
            sess.cond(c).duration = d;
            sess.cond(c).tmod = 0;
            sess.cond(c).pmod = struct('name', {}, 'param', {}, 'poly', {});
            sess.cond(c).orth = 0;
            c=c+1;
        end

        % Condition X+1: OUTCOMES - same stimuli as TOI
        [o, d] = get_onsets_except(sub, run, onsetFolder, out_name, trial_id, invert_flag); 
        sess.cond(c).name = sprintf('outcomes_cond-%s', out_name);
        sess.cond(c).onset = o;
        sess.cond(c).duration = d;
        sess.cond(c).tmod = 0;
        sess.cond(c).pmod = struct('name', {}, 'param', {}, 'poly', {});
        sess.cond(c).orth = 0;
        c=c+1;

       % Condition X+2-Z: CUES - OUTCOMES stimuli as TOI 
        out_cond_names = {'reward-high', 'reward-medium', 'reward-low','reward-hidden'};
        mask = ~strcmp(out_cond_names, out_name);
        other_conds_out = out_cond_names(mask);
        for i=1:length(other_conds_out)
            o_name = other_conds_out{i};
            [o, d] = get_onsets_except(sub, run, onsetFolder, o_name, [], invert_flag);
            sess.cond(c).name = sprintf('outcomes_cond-%s', o_name);
            sess.cond(c).onset = o;
            sess.cond(c).duration = d;
            sess.cond(c).tmod = 0;
            sess.cond(c).pmod = struct('name', {}, 'param', {}, 'poly', {});
            sess.cond(c).orth = 0;
            c=c+1;
        end

        % --- Motion regressors (24 from fMRIPrep)
        motion_vars = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z', ...
                       'trans_x_derivative1','trans_y_derivative1','trans_z_derivative1', ...
                       'rot_x_derivative1','rot_y_derivative1','rot_z_derivative1', ...
                       'trans_x_power2','trans_y_power2','trans_z_power2', ...
                       'rot_x_power2','rot_y_power2','rot_z_power2', ...
                       'trans_x_derivative1_power2','trans_y_derivative1_power2','trans_z_derivative1_power2', ...
                       'rot_x_derivative1_power2','rot_y_derivative1_power2','rot_z_derivative1_power2'};
        sess.regress = struct('name', {}, 'val', {}); 
        rcount = 0;
        for k = 1:numel(motion_vars)
            if ismember(motion_vars{k}, rp.Properties.VariableNames)
                rcount = rcount + 1;
                sess.regress(rcount).name = motion_vars{k};
                sess.regress(rcount).val  = rp.(motion_vars{k})';
            end
        end

        % --- Framewise displacement
        if ismember('framewise_displacement', rp.Properties.VariableNames)
            rcount = rcount + 1;
            fd = rp.framewise_displacement;
            fd(isnan(fd)) = 0;
            sess.regress(rcount).name = 'framewise_displacement';
            sess.regress(rcount).val = fd';
        end
        
        % --- Outlier regressors (motion_outlier*)
        outlier_vars = startsWith(rp.Properties.VariableNames, 'motion_outlier');
        outlier_names = rp.Properties.VariableNames(outlier_vars);
        for i = 1:numel(outlier_names)
            vals = rp.(outlier_names{i});
            if isempty(vals) || all(vals == 0)
                continue
            end
            rcount = rcount + 1;
            sess.regress(rcount).name = outlier_names{i};
            sess.regress(rcount).val = vals';
        end

        sess.multi_reg = {''};
        sess.hpf = 128;
        matlabbatch{1}.spm.stats.fmri_spec.sess(counter) = sess;
    end

    % Model settings
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];                                        
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0.1;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {fullfile(spm_path ,'tpm/mask_ICV.nii')}; 
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    % Run job
    spm_jobman('run', matlabbatch); 
    clear matlabbatch
end

function estimate_from_SPM_design_matrix(out_dir)
    % Estimates GLM parameters from a specified design matrix.
    matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
    matlabbatch{1}.spm.stats.fmri_est.spmmat = {fullfile(out_dir,'SPM.mat')};
    spm_jobman('run',matlabbatch);
    clear matlabbatch
end

function zip_spm_outputs(out_dir)
    % Gzip all NIfTI outputs in out_dir after estimation to save disk space.
    % SPM.mat is left uncompressed (the skip-if-done check reads it directly).
    nii_files = dir(fullfile(out_dir, '*.nii'));
    for i = 1:numel(nii_files)
        nii = fullfile(nii_files(i).folder, nii_files(i).name);
        gzip(nii);
        delete(nii);
    end
end