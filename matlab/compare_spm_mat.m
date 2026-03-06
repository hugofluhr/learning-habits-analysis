% Compare condition names across sessions in two SPM.mat files

% ---- INPUT ----
spmPath1 = '/Users/hugofluhr/phd_local/data/misc/Pakita-ppi/SPM_good.mat';
spmPath2 = '/Users/hugofluhr/phd_local/data/misc/Pakita-ppi/SPM_bug.mat';

% ---- LOAD SPM STRUCTURES ----
a = load(spmPath1, 'SPM');
b = load(spmPath2, 'SPM');

SPM1 = a.SPM;
SPM2 = b.SPM;

% ---- EXTRACT CONDITIONS BY SESSION ----
nSess1 = numel(SPM1.Sess);
nSess2 = numel(SPM2.Sess);

conds1 = cell(1, nSess1);
conds2 = cell(1, nSess2);

fprintf('=============================\n');
fprintf('EMPTY CONDITION CHECK: FILE 1\n');
fprintf('=============================\n');

for s = 1:nSess1
    U = SPM1.Sess(s).U;
    names = cell(1, numel(U));

    fprintf('\nSession %d\n', s);

    for k = 1:numel(U)
        if iscell(U(k).name)
            condName = U(k).name{1};
        else
            condName = U(k).name;
        end
        names{k} = condName;

        hasOnsets = isfield(U(k), 'ons') && ~isempty(U(k).ons);
        hasDurations = isfield(U(k), 'dur') && ~isempty(U(k).dur);

        if hasOnsets && hasDurations
            nOnsets = numel(U(k).ons);
            nDurations = numel(U(k).dur);

            if nOnsets == 0 || nDurations == 0
                fprintf('  [EMPTY] %s\n', condName);
            else
                fprintf('  [OK]    %s  (onsets=%d, durations=%d)\n', condName, nOnsets, nDurations);
            end
        else
            fprintf('  [EMPTY] %s  (missing onsets and/or durations)\n', condName);
        end
    end

    conds1{s} = names;
end

fprintf('\n=============================\n');
fprintf('EMPTY CONDITION CHECK: FILE 2\n');
fprintf('=============================\n');

for s = 1:nSess2
    U = SPM2.Sess(s).U;
    names = cell(1, numel(U));

    fprintf('\nSession %d\n', s);

    for k = 1:numel(U)
        if iscell(U(k).name)
            condName = U(k).name{1};
        else
            condName = U(k).name;
        end
        names{k} = condName;

        hasOnsets = isfield(U(k), 'ons') && ~isempty(U(k).ons);
        hasDurations = isfield(U(k), 'dur') && ~isempty(U(k).dur);

        if hasOnsets && hasDurations
            nOnsets = numel(U(k).ons);
            nDurations = numel(U(k).dur);

            if nOnsets == 0 || nDurations == 0
                fprintf('  [EMPTY] %s\n', condName);
            else
                fprintf('  [OK]    %s  (onsets=%d, durations=%d)\n', condName, nOnsets, nDurations);
            end
        else
            fprintf('  [EMPTY] %s  (missing onsets and/or durations)\n', condName);
        end
    end

    conds2{s} = names;
end

fprintf('\n=============================\n');
fprintf('CONDITION NAME COMPARISON\n');
fprintf('=============================\n');

fprintf('\nFile 1: %s\n', spmPath1);
fprintf('Sessions: %d\n', nSess1);

fprintf('\nFile 2: %s\n', spmPath2);
fprintf('Sessions: %d\n\n', nSess2);

nSess = min(nSess1, nSess2);

for s = 1:nSess
    c1 = conds1{s};
    c2 = conds2{s};

    fprintf('=== Session %d ===\n', s);

    only1 = setdiff(c1, c2);
    only2 = setdiff(c2, c1);
    common = intersect(c1, c2);

    fprintf('Common conditions (%d):\n', numel(common));
    disp(common')

    fprintf('Only in file 1 (%d):\n', numel(only1));
    disp(only1')

    fprintf('Only in file 2 (%d):\n', numel(only2));
    disp(only2')

    fprintf('\n');
end

if nSess1 ~= nSess2
    fprintf('WARNING: different number of sessions (%d vs %d)\n', nSess1, nSess2);
end