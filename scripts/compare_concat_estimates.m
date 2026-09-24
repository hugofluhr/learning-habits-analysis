% Compare the concatenated model (A) with the per-session model (B) for one subject, at the estimate level.
%
% Injected: dirA (concat), dirB (per-session) - subject folders containing SPM.mat, con_*.nii, spmT_*.nii,
% ResMS.nii and mask.nii. Prints regressor counts, residual df, ResMS, and per contrast the correlation of
% the t-maps and con images over the common mask.
%
% Contrasts are matched by index: the extra contrasts of the per-session model carry the " - All Sessions"
% suffix from sessrep 'repl'. Base contrasts in the per-session model sum three session betas (weight 1 each),
% so con images are on a different scale; the slope column shows con_B ~ slope * con_A.
% Caution: response and purple_frame can be near-collinear in the per-session model, which makes their
% per-session contrasts mostly noise (see check_concat_response_betas.py). Do not read their
% disagreement as a concat problem.
addpath("/home/hfluhr/repos/spm12");
rd = @(f) spm_read_vols(spm_vol(char(f)));
sa = load(fullfile(dirA, "SPM.mat")); sa = sa.SPM; sb = load(fullfile(dirB, "SPM.mat")); sb = sb.SPM;
fprintf("regressors: concat=%d  per-session=%d\n", size(sa.xX.X,2), size(sb.xX.X,2));
fprintf("residual df (xX.erdf): concat=%.1f  per-session=%.1f\n", sa.xX.erdf, sb.xX.erdf);

ma = rd(fullfile(dirA, "mask.nii")) > 0; mb = rd(fullfile(dirB, "mask.nii")) > 0;
m = ma & mb;
fprintf("mask voxels: concat=%d per-session=%d intersection=%d\n", nnz(ma), nnz(mb), nnz(m));

ra = rd(fullfile(dirA, "ResMS.nii")); rb = rd(fullfile(dirB, "ResMS.nii"));
fprintf("ResMS (mean in mask): concat=%.4g per-session=%.4g  corr across voxels=%.3f\n", mean(ra(m)), mean(rb(m)), corr(ra(m), rb(m)));

names = {sa.xCon.name}; assert(numel(names) == numel(sb.xCon), "different number of contrasts"); % names differ only by the " - All Sessions" suffix on repl contrasts
fprintf("\n%-20s %8s %8s %8s %8s\n", "contrast", "r(T)", "r(con)", "slope", "meanT A/B");
for i = 1:numel(names)
    ta = rd(fullfile(dirA, sprintf("spmT_%04d.nii", i))); tb = rd(fullfile(dirB, sprintf("spmT_%04d.nii", i)));
    ca = rd(fullfile(dirA, sprintf("con_%04d.nii", i))); cb = rd(fullfile(dirB, sprintf("con_%04d.nii", i)));
    ok = m & isfinite(ta) & isfinite(tb) & isfinite(ca) & isfinite(cb);
    slope = ca(ok) \ cb(ok); % con_B ~ slope * con_A (3 expected where per-session weights sum over 3 sessions)
    fprintf("%-20s %8.3f %8.3f %8.2f %4.2f/%4.2f\n", names{i}, corr(ta(ok), tb(ok)), corr(ca(ok), cb(ok)), slope, mean(ta(ok)), mean(tb(ok)));
end
