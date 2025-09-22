function export_first_lvl_contrasts(firstlvl_root, outdir, varargin)
% Export/alias SPM first-level contrasts with stable names for second-level use.
% DEFAULT: create readable symlinks IN PLACE beside each con_XXXX.nii
%
% - Splits SPM.mat files into 'learning' vs 'test' phases by subdirectory name.
% - Verifies each phase shares the same ordered list of contrast names (strict).
% - Writes per-folder manifests next to each SPM.mat.
% - Writes per-phase contrast order files at the project root.
% - By default creates symlinks (space-efficient); set 'copy', true to copy.
% - If you pass a non-empty outdir, it will export under outdir/<phase>/...
%
% Usage:
%   export_first_lvl_contrasts(firstlvl_root, '')                % in-place symlinks (default)
%   export_first_lvl_contrasts(firstlvl_root, '', 'copy', true)  % in-place copies
%   export_first_lvl_contrasts(firstlvl_root, '/path/to/out')    % export to outdir (per-contrast dirs)
%
% Options:
%   'pattern'        : glob to find SPM.mat (default '**/SPM.mat')
%   'copy'           : true to copy instead of symlink (default false)
%   'perContrastDirs': if exporting to outdir, split per contrast (default true)

p = inputParser;
addParameter(p, 'pattern', '**/SPM.mat', @ischar);
addParameter(p, 'perContrastDirs', true, @islogical);
addParameter(p, 'copy', false, @islogical);
parse(p, varargin{:});
opts = p.Results;

in_place = isempty(outdir) || strcmp(outdir, '');
if ~in_place
    if ~exist(outdir, 'dir'), mkdir(outdir); end
end

% Find all SPM.mat files
spms = dir(fullfile(firstlvl_root, opts.pattern));
spms = spms(~[spms.isdir]);
if isempty(spms)
    error('No SPM.mat files found under %s with pattern %s.', firstlvl_root, opts.pattern);
end

% Split into phases by path tokens (case-insensitive match on dir names)
is_learning = false(numel(spms),1);
is_test     = false(numel(spms),1);
for k = 1:numel(spms)
    f = lower(spms(k).folder);
    toks = regexp(f, filesep, 'split');
    is_learning(k) = any(strcmp(toks, 'learning'));
    is_test(k)     = any(strcmp(toks, 'test'));
end

spms_learning = spms(is_learning);
spms_test     = spms(is_test);
spms_other    = spms(~(is_learning | is_test));

if ~isempty(spms_other)
    fprintf(2, 'Warning: %d SPM.mat not in "learning" or "test" subdirs; skipping.\n', numel(spms_other));
    for k = 1:numel(spms_other)
        fprintf(2, '  %s\n', fullfile(spms_other(k).folder, spms_other(k).name));
    end
end

% Process each phase identically (strict within-phase checks)
if ~isempty(spms_learning)
    process_phase('learning', spms_learning, firstlvl_root, outdir, in_place, opts);
end
if ~isempty(spms_test)
    process_phase('test', spms_test, firstlvl_root, outdir, in_place, opts);
end

if isempty(spms_learning) && isempty(spms_test)
    error('Found SPM.mat files, but none under "learning" or "test" subdirectories.');
end

end % main function


% ---------- helpers ----------

function process_phase(phase_name, spms, firstlvl_root, outdir, in_place, opts)
% Helper: run the original pipeline on a subset (one phase)

sanitize = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-z0-9._-]', '');
strip_all_sessions = @(s) regexprep(s, '^\s*All Sessions\s*-\s*', '', 'ignorecase');

if ~in_place
    % Phase-specific export base to avoid mixing outputs
    outdir_phase = fullfile(outdir, phase_name);
    if ~exist(outdir_phase, 'dir'), mkdir(outdir_phase); end
else
    outdir_phase = ''; % unused
end

manifest_rows_global = {}; % only used when exporting to outdir
ref_names = string.empty(1,0);

for k = 1:numel(spms)
    spm_path = fullfile(spms(k).folder, spms(k).name);
    S = load(spm_path, 'SPM');
    SPM = S.SPM;

    % Build a subject/run token from relative path for manifests/exports
    rel = strrep(spm_path, [firstlvl_root filesep], '');
    parts = regexp(rel, filesep, 'split');
    token_parts = parts(1:max(1, numel(parts)-1)); % everything up to, but excluding, SPM.mat
    subj_token = sanitize(strjoin(token_parts, '_'));

    % Contrast names for this SPM (strip "All Sessions - ")
    xCon  = SPM.xCon(:);
    names = string({xCon.name});
    names = arrayfun(strip_all_sessions, names);

    % Validate name/order across all SPMs in this phase
    if isempty(ref_names)
        ref_names = names;
    else
        if numel(names) ~= numel(ref_names) || any(names ~= ref_names)
            fprintf(2, '\n[%s] Mismatch in contrast definitions for %s\n', upper(phase_name), spm_path);
            fprintf(2, 'Reference:\n  %s\n', strjoin(ref_names, ' | '));
            fprintf(2, 'This SPM:\n  %s\n', strjoin(names, ' | '));
            error('[%s] Contrast name/order mismatch. Fix first-level definitions before phase stats.', upper(phase_name));
        end
    end

    % Where to put outputs for this SPM (either in-place or under outdir/<phase>)
    src_dir = spms(k).folder;

    % Per-folder manifest if in-place; otherwise we'll also write a global one
    if in_place
        mf_path = fullfile(src_dir, 'contrast_manifest.tsv');
        fid = fopen(mf_path, 'w');
        fprintf(fid, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
    end

    % Process contrasts
    for i = 1:numel(xCon)
        cname   = char(ref_names(i));
        cname_s = sanitize(cname);

        % locate source con image
        if isfield(xCon(i), 'Vcon') && ~isempty(xCon(i).Vcon) && isfield(xCon(i).Vcon, 'fname')
            con_src = xCon(i).Vcon.fname;
            if ~isfile(con_src), con_src = fullfile(src_dir, sprintf('con_%04d.nii', i)); end
        else
            con_src = fullfile(src_dir, sprintf('con_%04d.nii', i));
        end
        if ~isfile(con_src)
            error('Missing contrast image: %s', con_src);
        end

        if in_place
            % In-place alias: con_XXXX_<name>.nii in the SAME directory
            con_dst = fullfile(src_dir, sprintf('con_%04d_%s.nii', i, cname_s));
            if opts.copy || ~isunix
                if ~isfile(con_dst), copyfile(con_src, con_dst); end
            else
                if ~isfile(con_dst)
                    system(sprintf('ln -s "%s" "%s"', con_src, con_dst));
                end
            end
            % Write row to the per-folder manifest
            fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', subj_token, i, cname, con_dst, con_src);

        else
            % Export to outdir/<phase>
            if opts.perContrastDirs
                dst_dir = fullfile(outdir_phase, sprintf('contrast-%02d_%s', i, cname_s));
            else
                dst_dir = outdir_phase;
            end
            if ~exist(dst_dir, 'dir'), mkdir(dst_dir); end
            dst_name = sprintf('%s_desc-%s_con.nii', subj_token, cname_s);
            con_dst  = fullfile(dst_dir, dst_name);

            if opts.copy || ~isunix
                if ~isfile(con_dst), copyfile(con_src, con_dst); end
            else
                if ~isfile(con_dst)
                    system(sprintf('ln -s "%s" "%s"', con_src, con_dst));
                end
            end
            % Collect for global manifest
            manifest_rows_global(end+1, :) = {subj_token, i, cname, con_dst, con_src}; %#ok<AGROW>
        end
    end

    if in_place
        fclose(fid);
        fprintf('[%s] Wrote in-place manifest: %s\n', upper(phase_name), mf_path);
    end
end

% Write reference contrast list (per phase) to appropriate location
if in_place
    % When in-place, write to first-level root for backwards compatibility
    order_path = fullfile(firstlvl_root, sprintf('contrast_list_order_phase-%s.txt', phase_name));
else
    % When exporting, write to output directory so second-level only needs export dir
    order_path = fullfile(outdir_phase, sprintf('contrast_list_order_phase-%s.txt', phase_name));
end

fid = fopen(order_path, 'w');
for i = 1:numel(ref_names)
    fprintf(fid, '%02d\t%s\n', i, ref_names(i));
end
fclose(fid);
fprintf('[%s] Reference contrast order: %s\n', upper(phase_name), order_path);

% If exporting to outdir, also write a global manifest there (per phase)
if ~in_place
    mf = fullfile(outdir_phase, 'contrasts_manifest.tsv');
    fid = fopen(mf, 'w');
    fprintf(fid, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
    for r = 1:size(manifest_rows_global,1)
        fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', manifest_rows_global{r,:});
    end
    fclose(fid);
    fprintf('[%s] Export complete. Global manifest: %s\n', upper(phase_name), mf);
else
    fprintf('[%s] In-place aliasing complete across %d first-level models.\n', upper(phase_name), numel(spms));
end

end % process_phase