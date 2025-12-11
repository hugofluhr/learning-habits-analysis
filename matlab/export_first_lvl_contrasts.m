function export_first_lvl_contrasts_auto(firstlvl_root, outdir, varargin)
% Export/alias SPM first-level contrasts for both classic (phase-split) and all-runs GLM outputs.
% Detects structure and processes accordingly.
% DEFAULT: create readable symlinks IN PLACE beside each con_XXXX.nii
% - Verifies all subjects share the same ordered list of contrast names (strict).
% - Writes per-folder manifests next to each SPM.mat.
% - Writes per-phase contrast order files if phase split, or a single order file if not.
% - By default creates symlinks (space-efficient); set 'copy', true to copy.
% - If you pass a non-empty outdir, it will export under outdir/...

% Usage:
%   export_first_lvl_contrasts_auto(firstlvl_root, '')                % in-place symlinks (default)
%   export_first_lvl_contrasts_auto(firstlvl_root, '', 'copy', true)  % in-place copies
%   export_first_lvl_contrasts_auto(firstlvl_root, '/path/to/out')    % export to outdir (per-contrast dirs)

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

sanitize = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-z0-9._-]', '');
strip_all_sessions = @(s) regexprep(s, '\s*-\s*All Sessions\s*$', '', 'ignorecase');

% Detect phase split structure by checking first subject
sub_dirs = dir(firstlvl_root);
sub_dirs = sub_dirs([sub_dirs.isdir] & ~startsWith({sub_dirs.name}, '.'));
phases = {'learning', 'test'};
has_phase = false;
if ~isempty(sub_dirs)
    subj_dir = fullfile(firstlvl_root, sub_dirs(1).name);
    for p = 1:numel(phases)
        phase = phases{p};
        phase_dir = fullfile(subj_dir, phase);
        if isfolder(phase_dir)
            has_phase = true;
        end
    end
end

if has_phase
    % --- PHASE SPLIT STRUCTURE (model_dir/sub/phase) ---
    for p = 1:numel(phases)
        phase = phases{p};
        manifest_rows_global = {};
        ref_names = string.empty(1,0);
        n_processed = 0;
        n_skipped   = 0;
        for s = 1:numel(sub_dirs)
            subj_dir = fullfile(firstlvl_root, sub_dirs(s).name);
            phase_dir = fullfile(subj_dir, phase);
            if ~isfolder(phase_dir)
                continue;
            end
            spms = dir(fullfile(phase_dir, opts.pattern));
            spms = spms(~[spms.isdir]);
            if isempty(spms)
                continue;
            end
            for k = 1:numel(spms)
                spm_path = fullfile(spms(k).folder, spms(k).name);
                S = load(spm_path, 'SPM');
                SPM = S.SPM;
                rel = strrep(spm_path, [subj_dir filesep phase filesep], '');
                parts = regexp(rel, filesep, 'split');
                token_parts = [{sub_dirs(s).name}, parts(1:max(1, numel(parts)-1))];
                subj_token = sanitize(strjoin(token_parts, '_'));
                if ~isfield(SPM, 'xCon') || isempty(SPM.xCon)
                    fprintf(2, '[%s] No contrasts defined in %s (only betas?) - skipping.\n', upper(phase), subj_token);
                    n_skipped = n_skipped + 1;
                    continue;
                end
                xCon  = SPM.xCon(:);
                names = string({xCon.name});
                names = arrayfun(strip_all_sessions, names);
                src_dir = spms(k).folder;
                first_con_guess = fullfile(src_dir, sprintf('con_%04d.nii', 1));
                if ~isfile(first_con_guess)
                    fprintf(2, '[%s] No contrast images found for %s (only betas?) - skipping.\n', upper(phase), subj_token);
                    n_skipped = n_skipped + 1;
                    continue;
                end
                if isempty(ref_names)
                    ref_names = names;
                else
                    if numel(names) ~= numel(ref_names) || any(names ~= ref_names)
                        fprintf(2, '\n[%s] Mismatch in contrast definitions for %s\n', upper(phase), spm_path);
                        fprintf(2, 'Reference:\n  %s\n', strjoin(ref_names, ' | '));
                        fprintf(2, 'This SPM:\n  %s\n', strjoin(names, ' | '));
                        error('[%s] Contrast name/order mismatch. Fix first-level definitions before phase stats.', upper(phase));
                    end
                end
                n_processed = n_processed + 1;
                if in_place
                    mf_path = fullfile(src_dir, 'contrast_manifest.tsv');
                    fid = fopen(mf_path, 'w');
                    fprintf(fid, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
                end
                for i = 1:numel(xCon)
                    cname   = char(ref_names(i));
                    cname_s = sanitize(cname);
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
                        con_dst = fullfile(src_dir, sprintf('con_%04d_%s.nii', i, cname_s));
                        if opts.copy || ~isunix
                            if ~isfile(con_dst), copyfile(con_src, con_dst); end
                        else
                            if ~isfile(con_dst)
                                system(sprintf('ln -s "%s" "%s"', con_src, con_dst));
                            end
                        end
                        fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', subj_token, i, cname, con_dst, con_src);
                    else
                        outdir_phase = fullfile(outdir, phase);
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
                        manifest_rows_global(end+1, :) = {subj_token, i, cname, con_dst, con_src}; %#ok<AGROW>
                    end
                end
                if in_place
                    fclose(fid);
                    fprintf('[%s] Wrote in-place manifest: %s\n', upper(phase), mf_path);
                end
            end
        end
        if in_place
            order_path = fullfile(phase_dir, sprintf('contrast_list_order_phase-%s.txt', phase));
        else
            order_path = fullfile(outdir, phase, sprintf('contrast_list_order_phase-%s.txt', phase));
        end
        fid = fopen(order_path, 'w');
        for i = 1:numel(ref_names)
            fprintf(fid, '%02d\t%s\n', i, ref_names(i));
        end
        fclose(fid);
        fprintf('[%s] Reference contrast order: %s\n', upper(phase), order_path);
        if ~in_place
            mf = fullfile(outdir, phase, 'contrasts_manifest.tsv');
            fid = fopen(mf, 'w');
            fprintf(fid, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
            for r = 1:size(manifest_rows_global,1)
                fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', manifest_rows_global{r,:});
            end
            fclose(fid);
            fprintf('[%s] Export complete. Global manifest: %s\n', upper(phase), mf);
        else
            fprintf('[%s] In-place aliasing complete across %d first-level models.\n', upper(phase), numel(spms));
        end
    end
else
    % --- ALL RUNS STRUCTURE ---
    spms = dir(fullfile(firstlvl_root, opts.pattern));
    spms = spms(~[spms.isdir]);
    if isempty(spms)
        error('No SPM.mat files found under %s with pattern %s.', firstlvl_root, opts.pattern);
    end
    if ~in_place
        outdir_export = outdir;
    else
        outdir_export = '';
    end
    manifest_rows_global = {};
    ref_names = string.empty(1,0);
    n_processed = 0;
    n_skipped   = 0;
    for k = 1:numel(spms)
        spm_path = fullfile(spms(k).folder, spms(k).name);
        S = load(spm_path, 'SPM');
        SPM = S.SPM;
        rel = strrep(spm_path, [firstlvl_root filesep], '');
        parts = regexp(rel, filesep, 'split');
        token_parts = parts(1:max(1, numel(parts)-1));
        subj_token = sanitize(strjoin(token_parts, '_'));
        if ~isfield(SPM, 'xCon') || isempty(SPM.xCon)
            fprintf(2, '[ALLRUNS] No contrasts defined in %s (only betas?) - skipping.\n', subj_token);
            n_skipped = n_skipped + 1;
            continue;
        end
        xCon  = SPM.xCon(:);
        names = string({xCon.name});
        names = arrayfun(strip_all_sessions, names);
        src_dir = spms(k).folder;
        first_con_guess = fullfile(src_dir, sprintf('con_%04d.nii', 1));
        if ~isfile(first_con_guess)
            fprintf(2, '[ALLRUNS] No contrast images found for %s (only betas?) - skipping.\n', subj_token);
            n_skipped = n_skipped + 1;
            continue;
        end
        if isempty(ref_names)
            ref_names = names;
        else
            if numel(names) ~= numel(ref_names) || any(names ~= ref_names)
                fprintf(2, '\n[ALLRUNS] Mismatch in contrast definitions for %s\n', spm_path);
                fprintf(2, 'Reference:\n  %s\n', strjoin(ref_names, ' | '));
                fprintf(2, 'This SPM:\n  %s\n', strjoin(names, ' | '));
                error('[ALLRUNS] Contrast name/order mismatch. Fix first-level definitions before group stats.');
            end
        end
        n_processed = n_processed + 1;
        if in_place
            mf_path = fullfile(src_dir, 'contrast_manifest.tsv');
            fid = fopen(mf_path, 'w');
            fprintf(fid, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
        end
        for i = 1:numel(xCon)
            cname   = char(ref_names(i));
            cname_s = sanitize(cname);
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
                con_dst = fullfile(src_dir, sprintf('con_%04d_%s.nii', i, cname_s));
                if opts.copy || ~isunix
                    if ~isfile(con_dst), copyfile(con_src, con_dst); end
                else
                    if ~isfile(con_dst)
                        system(sprintf('ln -s "%s" "%s"', con_src, con_dst));
                    end
                end
                fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', subj_token, i, cname, con_dst, con_src);
            else
                if opts.perContrastDirs
                    dst_dir = fullfile(outdir_export, sprintf('contrast-%02d_%s', i, cname_s));
                else
                    dst_dir = outdir_export;
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
                manifest_rows_global(end+1, :) = {subj_token, i, cname, con_dst, con_src}; %#ok<AGROW>
            end
        end
        if in_place
            fclose(fid);
            fprintf('[ALLRUNS] Wrote in-place manifest: %s\n', mf_path);
        end
    end
    if in_place
        order_path = fullfile(firstlvl_root, 'contrast_list_order_allruns.txt');
    else
        order_path = fullfile(outdir_export, 'contrast_list_order_allruns.txt');
    end
    fid = fopen(order_path, 'w');
    for i = 1:numel(ref_names)
        fprintf(fid, '%02d\t%s\n', i, ref_names(i));
    end
    fclose(fid);
    fprintf('[ALLRUNS] Reference contrast order: %s\n', order_path);
    if ~in_place
        mf = fullfile(outdir_export, 'contrasts_manifest.tsv');
        fid = fopen(mf, 'w');
        fprintf(fid, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
        for r = 1:size(manifest_rows_global,1)
            fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', manifest_rows_global{r,:});
        end
        fclose(fid);
        fprintf('[ALLRUNS] Export complete. Global manifest: %s\n', mf);
    else
        fprintf('[ALLRUNS] In-place aliasing complete across %d first-level models.\n', numel(spms));
    end
end