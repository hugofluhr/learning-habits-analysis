function svc_report(second_lvl_root, out_csv, mask_dir, spmpath)
% Peak table for every second-level model under second_lvl_root: whole brain and small-volume
% correction (SVC) in each ROI mask, as the SPM results GUI would give them.
%
%   svc_report(second_lvl_root, out_csv)
%   svc_report(second_lvl_root, out_csv, mask_dir, spmpath)
%
% Every T contrast of every SPM.mat found (recursively) is thresholded at p < 0.001 uncorrected,
% no extent threshold. SVC follows spm_VOI ('Image' search volume), but calls spm_list('Table')
% so it runs without graphics. One CSV row per local maximum; a region with no suprathreshold
% voxel gets one row with NaN statistics, so "nothing survives" is explicit.

if nargin < 3 || isempty(mask_dir)
    mask_dir = '/home/hfluhr/data/learninghabits/masks/MNI152NLin2009cAsym';
end
if nargin < 4 || isempty(spmpath)
    spmpath = '/home/hfluhr/repos/spm12';
end
if isempty(which('spm'))
    addpath(spmpath);
end
spm('defaults', 'fmri');
spm_get_defaults('cmdline', true);

masks = {
    'striatum_bartra', 'striatum_bartra2013_MNI152NLin2009cAsym.nii'
    'vmpfc_bartra',    'vmpfc_bartra2013_MNI152NLin2009cAsym.nii'
    'guida',           'habit_Guida2022_MNI152NLin2009cAsym.nii'
    'putamen_aal',     'putamen_AAL_MNI152NLin2009cAsym.nii'
    'motor_hmat',      'motor_HMAT_MNI152NLin2009cAsym.nii'
    'm1_hmat',         'motor_M1only_HMAT_MNI152NLin2009cAsym.nii'
    'premotor_hmat',   'premotor_HMAT_MNI152NLin2009cAsym.nii'
    'parietal_aal',    'parietal_AAL_MNI152NLin2009cAsym.nii'
};

% Local maxima listed: SPM's defaults for the whole-brain table and for SVC tables
wb_num = spm_get_defaults('stats.results.volume.nbmax');
wb_dis = spm_get_defaults('stats.results.volume.distmin');
svc_num = spm_get_defaults('stats.results.svc.nbmax');
svc_dis = spm_get_defaults('stats.results.svc.distmin');

files = dir(fullfile(second_lvl_root, '**', 'SPM.mat'));
rows = {};
for f = 1:numel(files)
    swd = files(f).folder;
    rel = strrep(swd, [second_lvl_root filesep], '');
    load(fullfile(swd, 'SPM.mat'), 'SPM');
    for ic = find(strcmp({SPM.xCon.STAT}, 'T'))
        xSPM = struct('swd', swd, 'title', '', 'Ic', ic, 'n', 1, 'Im', [], 'pm', [], 'Ex', [], ...
            'u', 0.001, 'thresDesc', 'none', 'k', 0);
        [SPMc, xSPM] = spm_getSPM(xSPM);
        subs = unique(regexp(strjoin(cellstr(SPMc.xY.P), ' '), 'sub-\d+', 'match'));
        base = {rel, ic, SPMc.xCon(ic).name, SPMc.nscan, strjoin(subs, ';')};

        TabDat = spm_list('Table', xSPM, wb_num, wb_dis);
        rows = [rows; table_rows(base, 'wholebrain', xSPM.S, TabDat)]; %#ok<AGROW>

        for m = 1:size(masks, 1)
            xS = svc(SPMc, xSPM, fullfile(mask_dir, masks{m, 2}));
            TabDat = spm_list('Table', xS, svc_num, svc_dis);
            rows = [rows; table_rows(base, masks{m, 1}, xS.S, TabDat)]; %#ok<AGROW>
        end
        fprintf('%s  [%d] %s  done\n', rel, ic, SPMc.xCon(ic).name);
    end
end

names = {'model', 'con_idx', 'con_name', 'n_subjects', 'subjects', 'region', 'search_voxels', ...
    'cluster_k', 'cluster_p_fwe', 'peak_p_fwe', 'peak_t', 'peak_z', 'x', 'y', 'z'};
writetable(cell2table(rows, 'VariableNames', names), out_csv);
fprintf('Wrote %s (%d rows)\n', out_csv, size(rows, 1));
end


function xSPM = svc(SPM, xSPM, mask_file)
% Restrict xSPM to a mask image and recompute the search volume, as spm_VOI does for 'I'.
XYZmm = SPM.xVol.M(1:3, :) * [SPM.xVol.XYZ; ones(1, SPM.xVol.S)];
D = spm_vol(mask_file);
VOX = sqrt(sum(D.mat(1:3, 1:3).^2));
FWHM = xSPM.FWHM .* (xSPM.VOX ./ VOX);
XYZ = D.mat \ [xSPM.XYZmm; ones(1, size(xSPM.XYZmm, 2))];
j = find(spm_sample_vol(D, XYZ(1, :), XYZ(2, :), XYZ(3, :), 0) > 0);
XYZ = D.mat \ [XYZmm; ones(1, size(XYZmm, 2))];
k = find(spm_sample_vol(D, XYZ(1, :), XYZ(2, :), XYZ(3, :), 0) > 0);

xSPM.S = length(k);
xSPM.R = spm_resels(FWHM, D, 'I');
xSPM.Z = xSPM.Z(j);
xSPM.XYZ = xSPM.XYZ(:, j);
xSPM.XYZmm = xSPM.XYZmm(:, j);
try, xSPM.Ps = xSPM.Ps(k); end
[up, xSPM.Pp] = spm_uc_peakFDR(0.05, xSPM.df, xSPM.STAT, xSPM.R, xSPM.n, xSPM.Vspm, k, xSPM.u);
uu = spm_uc(0.05, xSPM.df, xSPM.STAT, xSPM.R, xSPM.n, xSPM.S);
try
    V2R = 1 / prod(xSPM.FWHM(xSPM.DIM > 1));
    [uc, xSPM.Pc, ue] = spm_uc_clusterFDR(0.05, xSPM.df, xSPM.STAT, xSPM.R, xSPM.n, xSPM.Vspm, k, V2R, xSPM.u);
catch
    uc = NaN; ue = NaN; xSPM.Pc = [];
end
xSPM.uc = [uu up ue uc];
end


function rows = table_rows(base, region, search_voxels, TabDat)
% TabDat.dat columns: set p, c | cluster p(FWE), p(FDR), k, p(unc) | peak p(FWE), p(FDR), T, Z, p(unc) | xyz
if isempty(TabDat.dat)
    rows = [base, {region, search_voxels, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN}];
    return
end
n = size(TabDat.dat, 1);
rows = cell(n, 15);
k_cur = NaN; pc_cur = NaN;
for i = 1:n
    d = TabDat.dat(i, :);
    % continuation rows (further peaks in the same cluster) leave the cluster columns empty
    if ~isempty(d{5}), k_cur = d{5}; pc_cur = num_or_nan(d{3}); end
    xyz = d{12};
    rows(i, :) = [base, {region, search_voxels, k_cur, pc_cur, num_or_nan(d{7}), d{9}, d{10}, xyz(1), xyz(2), xyz(3)}];
end
end


function v = num_or_nan(v)
if isempty(v), v = NaN; end
end
