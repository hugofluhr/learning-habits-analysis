% add_session_contrasts_glm2_allruns_pmod_patch.m
%
% Appends the missing pmod per-session contrasts to glm2_all_runs.
% The four non-pmod session contrasts (first_stim, second_stim, response,
% purple_frame) were already added by add_session_contrasts_glm2.m.
% This patch adds the 4 missing single-regressor pmod contrasts per session:
% second_stimxQval, second_stimxHval, first_stimxQval, first_stimxHval.
% (Qval_diff/sum, Hval_diff/sum are cross-session and have no per-session form.)
%
% Safe to re-run: skips any contrast that already exists by name.

spmpath = '/home/ubuntu/repos/spm12';
if ~exist('glm_root', 'var') || isempty(glm_root)
    glm_root = '/mnt/data/learning-habits/spm_format/outputs/glm2_all_runs_scrubbed_2025-12-11-12-44';
end

addpath(spmpath);

% Note: Qval_diff/Hval_diff/Qval_sum/Hval_sum are cross-session combinations
% and have no per-session equivalent. Only the four single-regressor pmod
% contrasts are added here.
connames = {
    'second_stimxQval', 'second_stimxHval', ...
    'first_stimxQval',  'first_stimxHval'
};

session_labels = {'Session 1', 'Session 2', 'Session 3'};

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

sub_dirs = dir(glm_root);
sub_dirs = sub_dirs([sub_dirs.isdir] & ~startsWith({sub_dirs.name}, '.'));

if isempty(sub_dirs)
    error('No subject directories found under %s', glm_root);
end

for sd = 1:numel(sub_dirs)
    sub_id   = sub_dirs(sd).name;
    spm_path = fullfile(glm_root, sub_id, 'SPM.mat');
    if ~isfile(spm_path)
        fprintf('[SKIP] No SPM.mat for %s\n', sub_id);
        continue;
    end

    load(spm_path, 'SPM');
    colnames       = SPM.xX.name;
    nCols          = numel(colnames);
    nSess          = numel(SPM.Sess);
    existing_names = string({SPM.xCon.name});

    matlabbatch_con                         = {};
    matlabbatch_con{1}.spm.stats.con.spmmat = {spm_path};
    matlabbatch_con{1}.spm.stats.con.delete = 0;
    cc = 0;

    for si = 1:nSess
        cols_s  = SPM.Sess(si).col;
        names_s = colnames(cols_s);
        for ci = 1:numel(connames)
            cname     = connames{ci};
            con_label = sprintf('%s - %s', cname, session_labels{si});

            if any(existing_names == string(con_label))
                continue;  % already present, skip silently
            end

            idx_local = find(contains(names_s, cname), 1);
            if isempty(idx_local)
                fprintf('  [%s] "%s" absent in %s - skipped.\n', sub_id, cname, session_labels{si});
                continue;
            end

            cc            = cc + 1;
            w             = zeros(1, nCols);
            w(cols_s(idx_local)) = 1;
            matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.name    = con_label;
            matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.weights = w;
            matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.sessrep = 'none';
        end
    end

    if cc == 0
        fprintf('[SKIP] No new pmod session contrasts to add for %s.\n', sub_id);
        continue;
    end

    spm_jobman('run', matlabbatch_con);
    fprintf('[DONE] %s: added %d pmod per-session contrasts.\n', sub_id, cc);
end

fprintf('\nAll subjects processed.\n');
