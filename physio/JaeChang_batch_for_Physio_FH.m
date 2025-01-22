%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fMRI physio for FH
% Edited by JC Kim 06.06.2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; warning off;


spmpath = '/Users/jaekim/matlabwork/spm12/';
behpath = '/Volumes/MyData/Data/UZH/Anna/Behavior';
datapath = '/Volumes/MyData/Data/UZH/Anna';
analysispath = '/Users/jaekim/GDrive/analysis/UZH/Anna';

addpath(spmpath);
subjTable = readtable(fullfile(analysispath,'subjlists_all.xlsx')); % updated JC Kim 29.03.2023


subjlist = subjTable.participant_id;
seslist = subjTable.session_id;
nSubj = length(subjlist);

load(fullfile(analysispath,'BehTable_FH.mat'))

%%
% for contrast
conname = {'Common','CommonFacePercent','Uncommon','UncommonFacePercent','Feedback'};
ncon = length(conname);

nMotionReg = 6; % 6 for rp, 24? for PhysIO, 24+@ for PhysIO+Sensoring

contrastCommonFacePercent = zeros(1,ncon); contrastCommonFacePercent(2) = 1;
contrastUncommonFacePercent = zeros(1,ncon); contrastUncommonFacePercent(4) = 1;
contrastEmpty = zeros(1,ncon);
contrastMotionReg = zeros(1,nMotionReg);


% conname = {'Face0025','Face0075','Face0125','Face0175','Face0225','Face0275','Face0325','Face0375','Face0425','Face0475',...
%     'Face0525','Face0575','Face0625','Face0675','Face0725','Face0775','Face0825','Face0875','Face0925','Face0975',...
%     'Feedback'};


nvol = 210;
ntrials = 80;
TR=2.33;
nslices = 40;
runs = 1:4; % or 1:4
stimduration = 1.3; feedbackduration = 1;


FD_threshold = 0.5;

colFHTapasFail = zeros(nSubj,1);

for s = 1:nSubj
    if subjTable.ignore(s), continue; end
    if subjTable.nbehfh(s) < 5, continue; end

    clear matlabbatch;
    ST = clock;
    tic;
    % inputpaths - Physio files
    currsesPath = fullfile(datapath, subjlist{s}, seslist{s});

    physioPath = fullfile(datapath, subjlist{s}, seslist{s},'func');

    for r = runs

        currPhysio = spm_select('FPList', physioPath, sprintf('^%s_%s_task-fh_acq-.*_rec-1_run-%d.*.log$',subjlist{s},seslist{s},r));
        if size(currPhysio,1)>1, currPhysio = currPhysio(end,:); end

        matlabbatch{r}.spm.tools.physio.save_dir = cellstr(physioPath);
        matlabbatch{r}.spm.tools.physio.log_files.vendor = 'Philips';
        matlabbatch{r}.spm.tools.physio.log_files.cardiac = cellstr(currPhysio);
        matlabbatch{r}.spm.tools.physio.log_files.respiration = cellstr(currPhysio);
        matlabbatch{r}.spm.tools.physio.log_files.scan_timing = [];
        matlabbatch{r}.spm.tools.physio.log_files.sampling_interval = [];
        matlabbatch{r}.spm.tools.physio.log_files.relative_start_acquisition = 0;
        matlabbatch{r}.spm.tools.physio.log_files.align_scan = 'last';
        matlabbatch{r}.spm.tools.physio.scan_timing.sqpar.Nslices = nslices;
        matlabbatch{r}.spm.tools.physio.scan_timing.sqpar.NslicesPerBeat = [];
        matlabbatch{r}.spm.tools.physio.scan_timing.sqpar.TR = TR;
        matlabbatch{r}.spm.tools.physio.scan_timing.sqpar.Ndummies = 5;
        matlabbatch{r}.spm.tools.physio.scan_timing.sqpar.Nscans = nvol;
        matlabbatch{r}.spm.tools.physio.scan_timing.sqpar.onset_slice = 1; % First slice as onset-slice
        matlabbatch{r}.spm.tools.physio.scan_timing.sqpar.time_slice_to_slice = [];
        matlabbatch{r}.spm.tools.physio.scan_timing.sqpar.Nprep = [];
        matlabbatch{1}.spm.tools.physio.scan_timing.sync.nominal = struct([]);
        % matlabbatch{r}.spm.tools.physio.scan_timing.sync.gradient_log.grad_direction = 'z';
        % matlabbatch{r}.spm.tools.physio.scan_timing.sync.gradient_log.zero = 0.4; % From the preliminary results
        % matlabbatch{r}.spm.tools.physio.scan_timing.sync.gradient_log.slice = 0.5; % From the preliminary results
        % matlabbatch{r}.spm.tools.physio.scan_timing.sync.gradient_log.vol = [];
        % matlabbatch{r}.spm.tools.physio.scan_timing.sync.gradient_log.vol_spacing = 0.06; % From the preliminary results
        matlabbatch{r}.spm.tools.physio.preproc.cardiac.modality = 'ECG';
        matlabbatch{r}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
        matlabbatch{r}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
        matlabbatch{r}.spm.tools.physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
        matlabbatch{r}.spm.tools.physio.model.output_multiple_regressors = [currPhysio(1:end-4) '_multiple_regressors.txt'];
        matlabbatch{r}.spm.tools.physio.model.output_physio = [currPhysio(1:end-4) '_multiple_regressors.mat'];
        matlabbatch{r}.spm.tools.physio.model.orthogonalise = 'none';
        matlabbatch{r}.spm.tools.physio.model.censor_unreliable_recording_intervals = false;
        matlabbatch{r}.spm.tools.physio.model.retroicor.yes.order.c = 3;
        matlabbatch{r}.spm.tools.physio.model.retroicor.yes.order.r = 4;
        matlabbatch{r}.spm.tools.physio.model.retroicor.yes.order.cr = 1;
        matlabbatch{r}.spm.tools.physio.model.rvt.yes.delays = 0;
        matlabbatch{r}.spm.tools.physio.model.hrv.yes.delays = 0;
        matlabbatch{r}.spm.tools.physio.model.noise_rois.no = struct([]);
        matlabbatch{r}.spm.tools.physio.model.movement.yes.file_realignment_parameters = cellstr(spm_select('FPList', physioPath, sprintf('^rp_a%s_%s_task-fh_acq.*_run-%d_bold.txt$',subjlist{s}, seslist{s},r)));
        matlabbatch{r}.spm.tools.physio.model.movement.yes.order = 6;
        matlabbatch{r}.spm.tools.physio.model.movement.yes.censoring_method = 'FD';
        matlabbatch{r}.spm.tools.physio.model.movement.yes.censoring_threshold = FD_threshold;
        matlabbatch{r}.spm.tools.physio.model.other.no = struct([]);
        matlabbatch{r}.spm.tools.physio.verbose.level = 0;
        matlabbatch{r}.spm.tools.physio.verbose.fig_output_file = [currPhysio(1:end-4) '_multiple_regressors.fig'];
        matlabbatch{r}.spm.tools.physio.verbose.use_tabs = true;
    end
    spm_jobman('run',matlabbatch);


    elapsedtime = toc;
    %%
    ET = clock;

    fprintf('=======================================================================\n');
    fprintf('    %s is preprocessed\n', subjlist{s});
    fprintf('    Started Time : %g-%g-%g  %g:%g:%d \n', round(ST));
    fprintf('        End Time : %g-%g-%g  %g:%g:%d \n', round(ET));
    fprintf('    Elapsed Time : %g min. (%g sec.)\n',elapsedtime/60, elapsedtime);
    fprintf('=======================================================================\n\n');

    close all;

end


