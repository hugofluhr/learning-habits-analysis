function sfnr_calc()

% This script will use image data to calculate temporal SNR for fMRI datasets.
%
% NOTE!
% All calculations here are done from the first to the 2nd last (i.e. end-1 in Matlab lingo) dynamics.
% This doesn't affect the SFNR or BOLD sensitivity maps but is done because sometimes we collect data where
% the last dynamic is a noise volume.
%
%
% Written by Lydia Hellrung, November 2021

% Adapted by EMcP July 2025, hfluhr September 2025

toolbox_path = toolboxdir('images');
addpath(genpath(toolbox_path))
% add SPM to path
addpath('~/code/spm25/')

% Paths
plot_save_dir = '/Users/hugofluhr/phd_local/data/LearningHabits/sfnr'; % virtual machine path
dataDir = '/Users/hugofluhr/Downloads/raw_data2'; % virtual machine path

subjList = dir([dataDir filesep 'SNS_MRI_*']);

for subi= 1:height(subjList) % CHANGE BACK
    subjFolder = subjList(subi).name;
    subj = string(extractBetween(subjFolder,12,17)); % extract the participant id (XXXXX)
    subjDir = fullfile(dataDir,subjFolder);


    for run_no = 1:3
        fprintf('\n SFNR %s run %i\n',subj, run_no);

        scan_name = dir([subjDir filesep sprintf('sn*_run_%is.nii',run_no)]);

        if isempty(scan_name)
            warning('Missing run %d', run_no)
            continue
        elseif isscalar(scan_name)
            scan_name = scan_name.name;
        else
            warning('More than one run %d', run_no)
            scan_name = {scan_name(:).name};
            scan_name = sort(scan_name);
            scan_name = char(scan_name(end)); % takes the most recent file of that run
        end

        scan_name = fullfile(subjDir, scan_name);

        % ---- Output dirs/files
        out_dir = fullfile(plot_save_dir, sprintf('%s', subj), sprintf('run-%d', run_no));
        if ~exist(out_dir, 'dir'); mkdir(out_dir); end
        f_png   = fullfile(out_dir, sprintf('SFNR_check_S%s_R%i.png', subj, run_no));
        f_real  = fullfile(out_dir, 'realigned.nii');
        f_det   = fullfile(out_dir, 'detrended.nii');  % T-1 frames
        f_sfnr  = fullfile(out_dir, 'sfnr.nii');
        f_motion= fullfile(out_dir, 'motion.tsv');

        %% Loading Image Data

        HDRs = spm_vol(scan_name); % SENSE = ??? # TODO check correct!
        i32ch_epi  = spm_read_vols(HDRs);

        %% Realigning the data
        disp('Realigning volumes of the time series (might take a minute or two)...')
        [optimizer, metric] = imregconfig('monomodal'); % Needed for imregtform later % Requires image processing toolbox

        QAData = i32ch_epi;
        rQAData = zeros(size(QAData)); % Pre-allocating data for realigned data set

        for i = 1:size(QAData,4)
            %display(['Realigning volume #' num2str(i)])
            TempTrans = imregtform(squeeze(QAData(:,:,:,i)), squeeze(QAData(:,:,:,1)),'rigid', optimizer, metric);
            rQAData(:,:,:,i) = imwarp(squeeze(QAData(:,:,:,i)),TempTrans,'OutputView', imref3d(size(squeeze(QAData(:,:,:,1)))));

            Trans(i,1) = TempTrans.T(4,1);
            Trans(i,2) = TempTrans.T(4,2);
            Trans(i,3) = TempTrans.T(4,3);
            Rot(i,1)   = TempTrans.T(1,2);
            Rot(i,2)   = TempTrans.T(1,3);
            Rot(i,3)   = TempTrans.T(2,3);
        end

        ri32ch_3p0_TE30 = rQAData;

        % Save realigned 4D
        % write_4d_nii(HDRs, ri32ch_3p0_TE30, f_real);

        % Also save motion as TSV
        M = [Trans, Rot];
        fid = fopen(f_motion,'w'); fprintf(fid, "trans_x\ttrans_y\ttrans_z\trot_xy\trot_xz\trot_yz\n");
        fclose(fid);
        writematrix(M, f_motion, 'FileType','text','Delimiter','tab','WriteMode','append');

        %% Detrending the REALIGNED data

        QAData = ri32ch_3p0_TE30; % Make sure it's the realigned data from above
        dQAData = zeros(size(QAData));

        for x = 1:size(QAData,1)
            %display(['Working on #' num2str(x) ' out of ' num2str(size(QAData,1))])
            for y = 1:size(QAData,2)
                for z = 1:size(QAData,3)
                    VoxData = squeeze(QAData(x,y,z,1:(size(QAData,4)-1)));
                    p = polyfit(1:(size(QAData,4)-1), VoxData',2);
                    yfit = polyval(p,1:(size(QAData,4)-1));
                    dQAData(x,y,z,1:(size(QAData,4)-1)) = VoxData' - yfit + mean(VoxData);
                end
            end
        end

        dri32ch_3p0_TE30 = dQAData;

        % Save detrended 4D
        % write_4d_nii(HDRs(1:end-1), dri32ch_3p0_TE30, f_det);

        %% Calculating temporal SNR from detrended  data

        % First the voxel-wise mean
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        mdri32ch_3p0_TE30 = mean(dri32ch_3p0_TE30(:,:,:,1:end-1),4);

        % Next the temporal fluctuations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        sdri32ch_3p0_TE30 = std(dri32ch_3p0_TE30(:,:,:,1:end-1),0,4);

        % Finally the SFNR
        %%%%%%%%%%%%%%%%%%
        SFNR_i32ch_3p0_TE30 = mdri32ch_3p0_TE30./sdri32ch_3p0_TE30;

        % Save SFNR 3D
        W = HDRs(1); W.fname = f_sfnr; W.n = [1 1];
        spm_write_vol(W, SFNR_i32ch_3p0_TE30);

        % %% Displaying final SFNR results
        % figure('name', 'Temporal SNR Results')
        % 
        % slices = [3,5,7,9,11,13,17,21,25,29,33,37];
        % 
        % for slice_ind = 1:length(slices)
        %     slice_i = slices(slice_ind);
        % 
        %     subplot(3,4,slice_ind),  imagesc(rot90(squeeze(SFNR_i32ch_3p0_TE30(:,:, slice_i, 1)),-1), [0 140]), title(sprintf('slice no %i', slice_i)), axis image
        % end
        % 
        % colormap jet
        % exportgraphics(gcf, f_png, 'Resolution', 800)
    end
    close all
end
end

% ---------- Helper to write 4D NIfTI with SPM ----------
% function write_4d_nii(Vref, Y, out_file)
% if ~isstruct(Vref), error('Vref must be SPM vol struct(s)'); end
% if isscalar(Vref), Vref = repmat(Vref, 1, size(Y,4)); end
% Vout = Vref(1);
% Vout.fname = out_file;
% Vout.n = [1 1];
% Vout.dt = [spm_type('float32') spm_platform('bigend')];
% Vout = spm_create_vol(Vout);
% for t = 1:size(Y,4)
%     Vout.n = [t 1];
%     spm_write_vol(Vout, Y(:,:,:,t));
% end
% end