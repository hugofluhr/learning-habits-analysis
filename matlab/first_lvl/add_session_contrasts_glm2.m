% add_session_contrasts_glm2.m
%
% Appends per-session t-contrasts to an already-estimated first-level directory
% (one SPM.mat per subject). For each condition name and each SPM session, a
% contrast named "<cname> - Session <N>" is added.
%   Session 1 = learning1
%   Session 2 = learning2
%   Session 3 = test
%
% By default the condition names come from the model itself: every existing
% t-contrast whose name is a regressor (e.g. 'second_stimxQval_chosen'), so
% combined contrasts like 'Qval_sum' are left out. Inject connames to override.
%
% Usage: inject glm_root (and optionally connames), then run.

if ~exist('spmpath', 'var') || isempty(spmpath)
    spmpath = '/home/hfluhr/repos/spm12';
end
if ~exist('glm_root', 'var') || isempty(glm_root)
    error('Set glm_root to the first-level output directory (contains sub-XX/SPM.mat).');
end

addpath(spmpath);

if ~exist('connames', 'var')
    connames = {};  % empty = take them from each model's own contrasts
end

% Columns of regressor <cname>: "Sn(N) <cname>*bf(1)", or "Sn(N) <cname>^1*bf(1)" for a pmod
col_idx = @(names, cname) find(~cellfun(@isempty, regexp(names, ...
    ['^Sn\(\d+\) ' regexptranslate('escape', cname) '(\^\d+)?\*bf\(1\)$'], 'once')));

% One label per SPM.Sess index (must match session order in the GLM)
session_labels = {'Session 1', 'Session 2', 'Session 3'};  % 1=learning1 2=learning2 3=test

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

    if isempty(connames)
        sub_connames = {};
        for k = 1:numel(SPM.xCon)
            if strcmp(SPM.xCon(k).STAT, 'T') && ~isempty(col_idx(colnames, SPM.xCon(k).name))
                sub_connames{end+1} = SPM.xCon(k).name; %#ok<SAGROW>
            end
        end
        if isempty(sub_connames)
            error('[%s] No single-regressor t-contrasts in SPM.mat; inject connames.', sub_id);
        end
    else
        sub_connames = connames;
    end
    for ci = 1:numel(sub_connames)
        if isempty(col_idx(colnames, sub_connames{ci}))
            error('[%s] "%s" matches no regressor in any session.', sub_id, sub_connames{ci});
        end
    end
    fprintf('[%s] Conditions: %s\n', sub_id, strjoin(sub_connames, ', '));

    matlabbatch_con                         = {};
    matlabbatch_con{1}.spm.stats.con.spmmat = {spm_path};
    matlabbatch_con{1}.spm.stats.con.delete = 0;  % append, keep existing contrasts
    cc = 0;

    for si = 1:nSess
        cols_s  = SPM.Sess(si).col;
        names_s = colnames(cols_s);
        for ci = 1:numel(sub_connames)
            cname     = sub_connames{ci};
            idx_local = col_idx(names_s, cname);
            if isempty(idx_local)
                % legitimately absent in some sessions, e.g. points_feedback in the test run
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
