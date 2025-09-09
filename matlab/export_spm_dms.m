function export_spm_dms(base_dir, varargin)
% export_spm_designs_bids(base_dir, ...)
%
% Traverses base_dir/sub-XX/run-YY/SPM.mat and, for each SPM.mat, writes:
%   - sub-XX_run-YY_design_matrix.csv
%   - sub-XX_run-YY_column_names.txt
% in the SAME directory as the SPM.mat
%
% Options:
%   'Overwrite'  (false by default)
%   'SpmMatName' ('SPM.mat' by default)

p = inputParser;
p.addRequired('base_dir', @(s) ischar(s) || isstring(s));
p.addParameter('Overwrite', false, @(x) islogical(x) && isscalar(x));
p.addParameter('SpmMatName', 'SPM.mat', @(s) ischar(s) || isstring(s));
p.parse(base_dir, varargin{:});
opts = p.Results;

base_dir = char(base_dir);
spm_name = char(opts.SpmMatName);

if ~isfolder(base_dir)
    error('Base directory does not exist: %s', base_dir);
end

spm_files = dir(fullfile(base_dir, '**', spm_name));

if isempty(spm_files)
    fprintf('[export_spm_designs_bids] No %s files found under %s\n', spm_name, base_dir);
    return;
end

fprintf('[export_spm_designs_bids] Found %d %s files under %s\n', numel(spm_files), spm_name, base_dir);

for k = 1:numel(spm_files)
    d = spm_files(k).folder;
    spm_path = fullfile(d, spm_files(k).name);

    % --- derive BIDS-style prefix from path
    % look for sub-XX and run-YY in path
    tokens = regexp(d, '(sub-[^/\\]+).*?(run-[^/\\]+)', 'tokens', 'once');
    if isempty(tokens)
        % fallback: just use folder name
        prefix = sprintf('dir-%03d', k);
    else
        prefix = sprintf('%s_%s', tokens{1}, tokens{2});
    end

    out_csv  = fullfile(d, [prefix '_design_matrix.csv']);
    out_cols = fullfile(d, [prefix '_column_names.txt']);

    if ~opts.Overwrite && exist(out_csv,'file') && exist(out_cols,'file')
        fprintf('  [%3d/%3d] Skip (exists): %s\n', k, numel(spm_files), d);
        continue;
    end

    try
        S = load(spm_path, 'SPM');
        X = full(S.SPM.xX.X);

        % Write design matrix
        writematrix(X, out_csv);

        % Write column names
        names = string(S.SPM.xX.name(:));
        fid = fopen(out_cols, 'w');
        for i = 1:numel(names)
            fprintf(fid, '%s\n', names(i));
        end
        fclose(fid);

        fprintf('  [%3d/%3d] Wrote: %s\n', k, numel(spm_files), d);
    catch ME
        fprintf(2, '  [%3d/%3d] ERROR in %s: %s\n', k, numel(spm_files), d, ME.message);
    end
end
end