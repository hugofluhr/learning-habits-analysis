% Define paths and parameters
bids_root = '/Volumes/g_econ_department$/projects/2024/nebe_fluhr_timokhov_tobler_learning_habits/data/bids_dataset';
subjects_list_path = '/Users/hugofluhr/phd_local/data/LearningHabits/participants_sne2024.tsv';

% Read subjects list
subjects_list = readtable(subjects_list_path, 'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', false);
subjects = subjects_list.Var1;  % Assuming the file has no headers

% Define task names and run numbers
ses_id = 'ses-1';
task_names = {'learning', 'learning', 'test'};  % Adjust if necessary
n_vols = [426, 426, 593];
runs = 1:3;

for s = 3:length(subjects)

    sub_id = sprintf('sub-%02d', subjects(s));
    fprintf('Processing subject %s', sub_id);

    % Define functional data directory
    func_dir = fullfile(bids_root, sub_id, ses_id, 'func');
    
    clear matlabbatch;
    
    % Process each run
    for i = 1:length(runs)
    
        run_number = runs(i);
        task_name = task_names{i};
    
        % Construct filenames following BIDS convention
        physio_file = fullfile(func_dir, ...
            sprintf('%s_%s_task-%s_run-%d_physio.log', sub_id, ses_id, task_name, run_number));
        bold_file = fullfile(func_dir, ...
            sprintf('%s_%s_task-%s_run-%d_bold.nii', sub_id, ses_id, task_name, run_number));
    
        % Check if files exist before processing
        if exist(physio_file, 'file') && exist(bold_file, 'file')    
            
            matlabbatch{i}.spm.tools.physio.log_files.vendor = 'Philips';
            matlabbatch{i}.spm.tools.physio.log_files.cardiac = cellstr(physio_file);
            matlabbatch{i}.spm.tools.physio.log_files.respiration = cellstr(physio_file);
            matlabbatch{i}.spm.tools.physio.log_files.scan_timing = {''};
            matlabbatch{i}.spm.tools.physio.log_files.sampling_interval = [];
            matlabbatch{i}.spm.tools.physio.log_files.relative_start_acquisition = 0;
            matlabbatch{i}.spm.tools.physio.log_files.align_scan = 'last';
            matlabbatch{i}.spm.tools.physio.scan_timing.sqpar.Nslices = 40;
            matlabbatch{i}.spm.tools.physio.scan_timing.sqpar.NslicesPerBeat = [];
            matlabbatch{i}.spm.tools.physio.scan_timing.sqpar.TR = 2.33384;
            matlabbatch{i}.spm.tools.physio.scan_timing.sqpar.Ndummies = 5;
            matlabbatch{i}.spm.tools.physio.scan_timing.sqpar.Nscans = n_vols(i);
            matlabbatch{i}.spm.tools.physio.scan_timing.sqpar.onset_slice = 1;
            matlabbatch{i}.spm.tools.physio.scan_timing.sqpar.time_slice_to_slice = [];
            matlabbatch{i}.spm.tools.physio.scan_timing.sqpar.Nprep = [];
            matlabbatch{i}.spm.tools.physio.scan_timing.sync.gradient_log.grad_direction = 'y';
            matlabbatch{i}.spm.tools.physio.scan_timing.sync.gradient_log.zero = 0.5;
            matlabbatch{i}.spm.tools.physio.scan_timing.sync.gradient_log.slice = 0.6;
            matlabbatch{i}.spm.tools.physio.scan_timing.sync.gradient_log.vol = [];
            matlabbatch{i}.spm.tools.physio.scan_timing.sync.gradient_log.vol_spacing = [];
            matlabbatch{i}.spm.tools.physio.preproc.cardiac.modality = 'ECG';
            matlabbatch{i}.spm.tools.physio.preproc.cardiac.filter.no = struct([]);
            matlabbatch{i}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
            matlabbatch{i}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
            matlabbatch{i}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.max_heart_rate_bpm = 90;
            matlabbatch{i}.spm.tools.physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
            matlabbatch{i}.spm.tools.physio.preproc.respiratory.filter.passband = [0.01 2];
            matlabbatch{i}.spm.tools.physio.preproc.respiratory.despike = false;
            matlabbatch{i}.spm.tools.physio.model.orthogonalise = 'none';
            matlabbatch{i}.spm.tools.physio.model.censor_unreliable_recording_intervals = false;
            matlabbatch{i}.spm.tools.physio.model.retroicor.yes.order.c = 3;
            matlabbatch{i}.spm.tools.physio.model.retroicor.yes.order.r = 4;
            matlabbatch{i}.spm.tools.physio.model.retroicor.yes.order.cr = 1;
            matlabbatch{i}.spm.tools.physio.model.rvt.no = struct([]);
            matlabbatch{i}.spm.tools.physio.model.hrv.no = struct([]);
            matlabbatch{i}.spm.tools.physio.model.noise_rois.no = struct([]);
            matlabbatch{i}.spm.tools.physio.model.movement.no = struct([]);
            matlabbatch{i}.spm.tools.physio.model.other.no = struct([]);
            matlabbatch{i}.spm.tools.physio.verbose.level = 0;
            matlabbatch{i}.spm.tools.physio.verbose.use_tabs = false;       
    
            % Define output directory in BIDS structure
            output_dir = fullfile(bids_root, 'derivatives', 'physIO', sub_id, ses_id, 'func');
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            matlabbatch{i}.spm.tools.physio.save_dir = cellstr(output_dir);
    
            % Define output files
            matlabbatch{i}.spm.tools.physio.model.output_multiple_regressors = fullfile(output_dir, ...
                sprintf('%s_%s_task-%s_run-%d_physio.tsv', sub_id, ses_id, task_name, run_number));
            matlabbatch{i}.spm.tools.physio.model.output_physio = fullfile(output_dir, ...
                sprintf('%s_%s_task-%s_run-%d_physio.mat', sub_id, ses_id, task_name, run_number));
            %matlabbatch{i}.spm.tools.physio.model.verbose.fig_output_file = fullfile(output_dir, ...
            %    sprintf('%s_%s_task-%s_run-%d_physio.fig', sub_id, ses_id, task_name, run_number));
    
        else
            fprintf('Skipping missing files for Task %s, Run %d\n', task_name, run_number);
        end
    end
    
    spm_jobman('run', matlabbatch);

end