% add_session_contrasts_ppi.m
%
% Appends per-session t-contrasts to an already-estimated gPPI model
% (PPI_putamen subdir). For each PPI regressor in connames and each SPM
% session, a contrast named "<cname> - Session <N>" is added.
%   Session 1 = learning1
%   Session 2 = learning2
%   Session 3 = test
%
% Usage: set gppi_root below and run, or inject via:
%   matlab -nodisplay -r "gppi_root = '...'; run('add_session_contrasts_ppi.m'); exit"

spmpath  = '/home/ubuntu/repos/spm12';
if ~exist('gppi_root', 'var') || isempty(gppi_root)
    gppi_root = '';  % <-- SET: full path to gppi_putamen_* directory
end

addpath(spmpath);

connames = {
    'PPI_first_stim', 'PPI_second_stim', ...
    'PPI_second_stimxHval_chosen^1', ...
    'PPI_response', 'PPI_purple_frame', 'PPI_points_feedback'
};

session_labels = {'Session 1', 'Session 2', 'Session 3'};  % 1=learning1 2=learning2 3=test

spm('Defaults', 'fMRI');
spm_jobman('initcfg');

all_entries = dir(gppi_root);
sub_dirs = all_entries([all_entries.isdir] & startsWith({all_entries.name}, 'sub-'));

if isempty(sub_dirs)
    error('No sub-* directories found under %s', gppi_root);
end

for sd = 1:numel(sub_dirs)
    sub_id   = sub_dirs(sd).name;
    spm_path = fullfile(gppi_root, sub_id, 'PPI_putamen', 'SPM.mat');
    if ~isfile(spm_path)
        fprintf('[SKIP] No PPI_putamen/SPM.mat for %s\n', sub_id);
        continue;
    end

    load(spm_path, 'SPM');
    colnames = SPM.xX.name;
    nCols    = numel(colnames);
    nSess    = numel(SPM.Sess);

    if nSess ~= numel(session_labels)
        warning('[%s] Has %d sessions but session_labels has %d entries - check configuration.', ...
            sub_id, nSess, numel(session_labels));
    end

    % Check that no per-session contrasts already exist to avoid duplicates
    existing_names = string({SPM.xCon.name});
    already_has_sess = any(~cellfun(@isempty, regexp(cellstr(existing_names), ...
        '\s*-\s*Session\s+\d+\s*$', 'once', 'ignorecase')));
    if already_has_sess
        fprintf('[SKIP] %s already has per-session contrasts - skipping to avoid duplicates.\n', sub_id);
        continue;
    end

    matlabbatch_con                         = {};
    matlabbatch_con{1}.spm.stats.con.spmmat = {spm_path};
    matlabbatch_con{1}.spm.stats.con.delete = 0;  % append, keep existing contrasts
    cc = 0;

    for si = 1:nSess
        cols_s  = SPM.Sess(si).col;
        names_s = colnames(cols_s);
        for ci = 1:numel(connames)
            cname     = connames{ci};
            idx_local = find(contains(names_s, cname), 1);
            if isempty(idx_local)
                fprintf('  [%s] "%s" absent in %s - skipped.\n', sub_id, cname, session_labels{si});
                continue;
            end
            cc            = cc + 1;
            w             = zeros(1, nCols);
            w(cols_s(idx_local)) = 1;
            con_label     = sprintf('%s - %s', cname, session_labels{si});
            matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.name    = con_label;
            matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.weights = w;
            matlabbatch_con{1}.spm.stats.con.consess{cc}.tcon.sessrep = 'none';
        end
    end

    if cc == 0
        fprintf('[SKIP] No per-session contrasts to add for %s.\n', sub_id);
        continue;
    end

    spm_jobman('run', matlabbatch_con);
    fprintf('[DONE] %s: added %d per-session contrasts.\n', sub_id, cc);
end

fprintf('\nAll subjects processed.\n');
