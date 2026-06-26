function export_first_lvl_contrasts_with_sessions(firstlvl_root, outdir, varargin)
% Export SPM first-level contrasts, separating all-runs averages from
% per-session contrasts (named "<name> - Session <N>").
%
% Behaves identically to export_first_lvl_contrasts_auto for all-runs average
% contrasts.  Per-session contrasts are routed to separate subdirectories:
%   outdir/allruns/   - all-runs average contrasts
%   outdir/session-01/, session-02/, session-03/ - per-session contrasts
%
% In in-place mode, per-session contrasts get symlinks named
%   con_XXXX_<cname>_sess-NN.nii  and a separate contrast_manifest_sessions.tsv
%
% Usage:
%   export_first_lvl_contrasts_with_sessions(firstlvl_root, '')
%   export_first_lvl_contrasts_with_sessions(firstlvl_root, '/path/to/out')
%   export_first_lvl_contrasts_with_sessions(firstlvl_root, '', 'copy', true)
%
% Options:
%   'pattern'        : glob to find SPM.mat (default '**/SPM.mat')
%   'copy'           : true to copy instead of symlink (default false)
%   'perContrastDirs': split output into per-contrast subdirs (default true)

p = inputParser;
addParameter(p, 'pattern',         '**/SPM.mat', @ischar);
addParameter(p, 'perContrastDirs', true,         @islogical);
addParameter(p, 'copy',            false,        @islogical);
parse(p, varargin{:});
opts = p.Results;

in_place = isempty(outdir) || strcmp(outdir, '');
if ~in_place && ~exist(outdir, 'dir'), mkdir(outdir); end

sanitize       = @(s) regexprep(lower(strrep(strtrim(s), ' ', '-')), '[^a-z0-9._-]', '');
strip_all_sess = @(s) regexprep(s, '\s*-\s*All Sessions\s*$', '', 'ignorecase');

% Detect phase-split structure
sub_dirs = dir(firstlvl_root);
sub_dirs = sub_dirs([sub_dirs.isdir] & ~startsWith({sub_dirs.name}, '.'));
phases    = {'learning', 'test'};
has_phase = false;
if ~isempty(sub_dirs)
    for ph = 1:numel(phases)
        if isfolder(fullfile(firstlvl_root, sub_dirs(1).name, phases{ph}))
            has_phase = true; break;
        end
    end
end

if has_phase
    export_phase_split(firstlvl_root, sub_dirs, phases, outdir, opts, in_place, sanitize, strip_all_sess);
else
    export_allruns_with_sessions(firstlvl_root, outdir, opts, in_place, sanitize, strip_all_sess);
end

end % main function


% =========================================================================
function export_allruns_with_sessions(firstlvl_root, outdir, opts, in_place, sanitize, strip_all_sess)

spms = dir(fullfile(firstlvl_root, opts.pattern));
spms = spms(~[spms.isdir]);
if isempty(spms)
    error('No SPM.mat files found under %s with pattern %s.', firstlvl_root, opts.pattern);
end

% Reference contrast lists (checked for consistency across subjects)
ref_allruns = string.empty(1,0);  % all-runs base names in order
ref_sess    = {};                  % ref_sess{N} = string array of base names for session N

% Global manifests (outdir mode only)
manifest_allruns = {};
manifest_sess    = {};

n_proc = 0; n_skip = 0;

for k = 1:numel(spms)
    spm_path   = fullfile(spms(k).folder, spms(k).name);
    S          = load(spm_path, 'SPM');
    SPM        = S.SPM;
    rel        = strrep(spm_path, [firstlvl_root filesep], '');
    parts      = regexp(rel, filesep, 'split');
    subj_token = sanitize(strjoin(parts(1:max(1, numel(parts)-1)), '_'));

    if ~isfield(SPM, 'xCon') || isempty(SPM.xCon)
        fprintf(2, '[ALLRUNS] No contrasts in %s - skipping.\n', subj_token);
        n_skip = n_skip + 1; continue;
    end

    src_dir = spms(k).folder;
    if ~isfile(fullfile(src_dir, sprintf('con_%04d.nii', 1)))
        fprintf(2, '[ALLRUNS] No contrast images for %s (only betas?) - skipping.\n', subj_token);
        n_skip = n_skip + 1; continue;
    end

    xCon = SPM.xCon(:);

    % ---- Split contrasts into all-runs vs per-session ----
    allruns_idx   = [];
    allruns_names = string.empty(1,0);
    sess_idx      = [];   % index into xCon
    sess_nums     = [];   % session number (from "- Session N")
    sess_base     = string.empty(1,0);  % base name without session suffix

    for i = 1:numel(xCon)
        raw = char(strip_all_sess(string(xCon(i).name)));
        tok = regexp(raw, '\s*-\s*Session\s+(\d+)\s*$', 'tokens', 'once', 'ignorecase');
        if ~isempty(tok)
            sn   = str2double(tok{1});
            base = strtrim(regexprep(raw, '\s*-\s*Session\s+\d+\s*$', '', 'ignorecase'));
            sess_idx(end+1)  = i;    %#ok<AGROW>
            sess_nums(end+1) = sn;   %#ok<AGROW>
            sess_base(end+1) = string(base); %#ok<AGROW>
        else
            allruns_idx(end+1)  = i;          %#ok<AGROW>
            allruns_names(end+1) = string(raw); %#ok<AGROW>
        end
    end

    % ---- Consistency checks ----
    if isempty(ref_allruns)
        ref_allruns = allruns_names;
    elseif numel(allruns_names) ~= numel(ref_allruns) || any(allruns_names ~= ref_allruns)
        fprintf(2, '\n[ALLRUNS] All-runs contrast mismatch for %s\n', spm_path);
        fprintf(2, 'Reference:\n  %s\n', strjoin(ref_allruns, ' | '));
        fprintf(2, 'This SPM:\n  %s\n', strjoin(allruns_names, ' | '));
        error('[ALLRUNS] Contrast name/order mismatch. Fix first-level definitions.');
    end

    for sn = unique(sess_nums)
        mask_sn  = sess_nums == sn;
        names_sn = sess_base(mask_sn);
        if numel(ref_sess) < sn || isempty(ref_sess{sn})
            ref_sess{sn} = names_sn;
        elseif numel(names_sn) ~= numel(ref_sess{sn}) || any(names_sn ~= ref_sess{sn})
            fprintf(2, '\n[SESSION-%02d] Contrast mismatch for %s\n', sn, spm_path);
            fprintf(2, 'Reference:\n  %s\n', strjoin(ref_sess{sn}, ' | '));
            fprintf(2, 'This SPM:\n  %s\n', strjoin(names_sn, ' | '));
            error('[SESSION-%02d] Contrast name/order mismatch.', sn);
        end
    end

    n_proc = n_proc + 1;

    % ---- Export all-runs contrasts ----
    if in_place
        fid_ar = fopen(fullfile(src_dir, 'contrast_manifest.tsv'), 'w');
        fprintf(fid_ar, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
    end

    for ii = 1:numel(allruns_idx)
        i       = allruns_idx(ii);
        cname   = char(ref_allruns(ii));
        cname_s = sanitize(cname);
        con_src = resolve_con_src(xCon, i, src_dir);
        if isempty(con_src), continue; end

        if in_place
            con_dst = fullfile(src_dir, sprintf('con_%04d_%s.nii', i, cname_s));
            link_or_copy(con_src, con_dst, opts.copy);
            fprintf(fid_ar, '%s\t%d\t%s\t%s\t%s\n', subj_token, i, cname, con_dst, con_src);
        else
            allruns_dir = fullfile(outdir, 'allruns');
            if opts.perContrastDirs
                dst_dir = fullfile(allruns_dir, sprintf('contrast-%02d_%s', ii, cname_s));
            else
                dst_dir = allruns_dir;
            end
            if ~exist(dst_dir, 'dir'), mkdir(dst_dir); end
            con_dst = fullfile(dst_dir, sprintf('%s_desc-%s_con.nii', subj_token, cname_s));
            link_or_copy(con_src, con_dst, opts.copy);
            manifest_allruns(end+1, :) = {subj_token, ii, cname, con_dst, con_src}; %#ok<AGROW>
        end
    end

    if in_place
        fclose(fid_ar);
        fprintf('[ALLRUNS] Wrote in-place manifest: %s\n', fullfile(src_dir, 'contrast_manifest.tsv'));
    end

    % ---- Export per-session contrasts ----
    if ~isempty(sess_idx)
        if in_place
            fid_ss = fopen(fullfile(src_dir, 'contrast_manifest_sessions.tsv'), 'w');
            fprintf(fid_ss, 'subject_token\tsession\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
        end

        for ii = 1:numel(sess_idx)
            i    = sess_idx(ii);
            sn   = sess_nums(ii);
            base = char(sess_base(ii));
            cname_s  = sanitize(base);
            con_src  = resolve_con_src(xCon, i, src_dir);
            if isempty(con_src), continue; end
            % Index of this contrast within its session's list
            si_within = sum(sess_nums(1:ii) == sn);

            if in_place
                % Encode session in filename: <cname>_sess-NN
                con_dst = fullfile(src_dir, sprintf('con_%04d_%s_sess-%02d.nii', i, cname_s, sn));
                link_or_copy(con_src, con_dst, opts.copy);
                fprintf(fid_ss, '%s\t%d\t%d\t%s\t%s\t%s\n', ...
                    subj_token, sn, si_within, base, con_dst, con_src);
            else
                sess_out = fullfile(outdir, sprintf('session-%02d', sn));
                if opts.perContrastDirs
                    dst_dir = fullfile(sess_out, sprintf('contrast-%02d_%s', si_within, cname_s));
                else
                    dst_dir = sess_out;
                end
                if ~exist(dst_dir, 'dir'), mkdir(dst_dir); end
                con_dst = fullfile(dst_dir, sprintf('%s_desc-%s_con.nii', subj_token, cname_s));
                link_or_copy(con_src, con_dst, opts.copy);
                manifest_sess(end+1, :) = {subj_token, sn, si_within, base, con_dst, con_src}; %#ok<AGROW>
            end
        end

        if in_place
            fclose(fid_ss);
            fprintf('[ALLRUNS] Wrote session manifest: %s\n', ...
                fullfile(src_dir, 'contrast_manifest_sessions.tsv'));
        end
    end
end % subject loop

% ---- Order files ----
% All-runs
if in_place
    order_path = fullfile(firstlvl_root, 'contrast_list_order_allruns.txt');
else
    ar_dir = fullfile(outdir, 'allruns');
    if ~exist(ar_dir, 'dir'), mkdir(ar_dir); end
    order_path = fullfile(ar_dir, 'contrast_list_order_allruns.txt');
end
fid = fopen(order_path, 'w');
for i = 1:numel(ref_allruns)
    fprintf(fid, '%02d\t%s\n', i, ref_allruns(i));
end
fclose(fid);
fprintf('[ALLRUNS] Contrast order: %s\n', order_path);

% Per-session
for sn = 1:numel(ref_sess)
    if isempty(ref_sess) || numel(ref_sess) < sn || isempty(ref_sess{sn})
        continue;
    end
    if in_place
        order_path = fullfile(firstlvl_root, sprintf('contrast_list_order_session-%02d.txt', sn));
    else
        ss_dir = fullfile(outdir, sprintf('session-%02d', sn));
        if ~exist(ss_dir, 'dir'), mkdir(ss_dir); end
        order_path = fullfile(ss_dir, sprintf('contrast_list_order_session-%02d.txt', sn));
    end
    fid = fopen(order_path, 'w');
    for i = 1:numel(ref_sess{sn})
        fprintf(fid, '%02d\t%s\n', i, ref_sess{sn}(i));
    end
    fclose(fid);
    fprintf('[SESSION-%02d] Contrast order: %s\n', sn, order_path);
end

% ---- Global manifests (outdir mode only) ----
if ~in_place
    ar_dir = fullfile(outdir, 'allruns');
    if ~exist(ar_dir, 'dir'), mkdir(ar_dir); end
    mf = fullfile(ar_dir, 'contrasts_manifest.tsv');
    fid = fopen(mf, 'w');
    fprintf(fid, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
    for r = 1:size(manifest_allruns, 1)
        fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', manifest_allruns{r,:});
    end
    fclose(fid);
    fprintf('[ALLRUNS] Global manifest: %s\n', mf);

    if ~isempty(manifest_sess)
        mf = fullfile(outdir, 'contrasts_manifest_sessions.tsv');
        fid = fopen(mf, 'w');
        fprintf(fid, 'subject_token\tsession\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
        for r = 1:size(manifest_sess, 1)
            fprintf(fid, '%s\t%d\t%d\t%s\t%s\t%s\n', manifest_sess{r,:});
        end
        fclose(fid);
        fprintf('[SESSIONS] Global manifest: %s\n', mf);
    end
end

fprintf('[ALLRUNS] Done. Processed %d, skipped %d.\n', n_proc, n_skip);
end % export_allruns_with_sessions


% =========================================================================
% Phase-split branch: identical logic to export_first_lvl_contrasts_auto.
% Per-session contrasts present in phase-split models are exported as-is
% (their sanitized names will include the session suffix).
function export_phase_split(firstlvl_root, sub_dirs, phases, outdir, opts, in_place, sanitize, strip_all_sess)

for ph = 1:numel(phases)
    phase = phases{ph};
    manifest_rows_global = {};
    ref_names  = string.empty(1,0);
    n_processed = 0;
    n_skipped   = 0;

    for s = 1:numel(sub_dirs)
        subj_dir  = fullfile(firstlvl_root, sub_dirs(s).name);
        phase_dir = fullfile(subj_dir, phase);
        if ~isfolder(phase_dir), continue; end

        spms = dir(fullfile(phase_dir, opts.pattern));
        spms = spms(~[spms.isdir]);
        if isempty(spms), continue; end

        for k = 1:numel(spms)
            spm_path = fullfile(spms(k).folder, spms(k).name);
            S = load(spm_path, 'SPM');
            SPM = S.SPM;
            rel   = strrep(spm_path, [subj_dir filesep phase filesep], '');
            parts = regexp(rel, filesep, 'split');
            token_parts = [{sub_dirs(s).name}, parts(1:max(1, numel(parts)-1))];
            subj_token  = sanitize(strjoin(token_parts, '_'));

            if ~isfield(SPM, 'xCon') || isempty(SPM.xCon)
                fprintf(2, '[%s] No contrasts in %s - skipping.\n', upper(phase), subj_token);
                n_skipped = n_skipped + 1; continue;
            end

            src_dir = spms(k).folder;
            if ~isfile(fullfile(src_dir, sprintf('con_%04d.nii', 1)))
                fprintf(2, '[%s] No contrast images for %s - skipping.\n', upper(phase), subj_token);
                n_skipped = n_skipped + 1; continue;
            end

            xCon  = SPM.xCon(:);
            names = arrayfun(@(c) strip_all_sess(string(c.name)), xCon);

            if isempty(ref_names)
                ref_names = names;
            elseif numel(names) ~= numel(ref_names) || any(names ~= ref_names)
                fprintf(2, '\n[%s] Mismatch for %s\nReference:\n  %s\nThis SPM:\n  %s\n', ...
                    upper(phase), spm_path, strjoin(ref_names, ' | '), strjoin(names, ' | '));
                error('[%s] Contrast name/order mismatch.', upper(phase));
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
                con_src = resolve_con_src(xCon, i, src_dir);

                if in_place
                    con_dst = fullfile(src_dir, sprintf('con_%04d_%s.nii', i, cname_s));
                    link_or_copy(con_src, con_dst, opts.copy);
                    fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', subj_token, i, cname, con_dst, con_src);
                else
                    outdir_phase = fullfile(outdir, phase);
                    if opts.perContrastDirs
                        dst_dir = fullfile(outdir_phase, sprintf('contrast-%02d_%s', i, cname_s));
                    else
                        dst_dir = outdir_phase;
                    end
                    if ~exist(dst_dir, 'dir'), mkdir(dst_dir); end
                    con_dst = fullfile(dst_dir, sprintf('%s_desc-%s_con.nii', subj_token, cname_s));
                    link_or_copy(con_src, con_dst, opts.copy);
                    manifest_rows_global(end+1, :) = {subj_token, i, cname, con_dst, con_src}; %#ok<AGROW>
                end
            end

            if in_place
                fclose(fid);
                fprintf('[%s] Wrote in-place manifest: %s\n', upper(phase), mf_path);
            end
        end
    end

    % Order file
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
    fprintf('[%s] Contrast order: %s\n', upper(phase), order_path);

    if ~in_place
        mf = fullfile(outdir, phase, 'contrasts_manifest.tsv');
        fid = fopen(mf, 'w');
        fprintf(fid, 'subject_token\tcontrast_index\tcontrast_name\tpath_dst\tpath_src\n');
        for r = 1:size(manifest_rows_global, 1)
            fprintf(fid, '%s\t%d\t%s\t%s\t%s\n', manifest_rows_global{r,:});
        end
        fclose(fid);
        fprintf('[%s] Export complete. Global manifest: %s\n', upper(phase), mf);
    else
        fprintf('[%s] In-place aliasing complete across %d first-level models.\n', upper(phase), n_processed);
    end
end
end % export_phase_split


% =========================================================================
function src = resolve_con_src(xCon, i, src_dir)
if isfield(xCon(i), 'Vcon') && ~isempty(xCon(i).Vcon) && isfield(xCon(i).Vcon, 'fname')
    src = xCon(i).Vcon.fname;
    if ~isfile(src)
        src = fullfile(src_dir, sprintf('con_%04d.nii', i));
    end
else
    src = fullfile(src_dir, sprintf('con_%04d.nii', i));
end
if ~isfile(src)
    fprintf(2, '[SKIP] Contrast %d has no image on disk (%s) - skipping.\n', i, src);
    src = '';
end
end


function link_or_copy(src, dst, do_copy)
if do_copy || ~isunix
    if ~isfile(dst), copyfile(src, dst); end
else
    if ~isfile(dst)
        system(sprintf('ln -s "%s" "%s"', src, dst));
    end
end
end
