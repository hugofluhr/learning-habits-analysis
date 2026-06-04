function [cond_name, stim_name, onset, duration] = get_trial_info(subject, run, bidsFolder, timepoint, trial_id, invert_flag)
% 
% Get spm parameters for a specific trial from bids/events file. 
% Usage:
%   [name, onset, duration] = get_trial_info(subject, bidsFolder, run, trial_id);
% Args:
%   subject: [string] 
%   bidsFolder: [string] ...path to bids/ dir (uses: bids/subject/events)
%   run: [int]
%   trial_id: [int] 
% Returns:
%   name: [string]
%   onset: [float]
%   duration: [float]    

    % --- Locate onset file ---
    eventsFolder = fullfile(bidsFolder, subject, "events");
    if ~isfolder(eventsFolder)
        error('Events folder not found: %s', eventsFolder);
    end
    pattern = sprintf("data_%s_*causal*.csv", subject);
    fileList = dir(fullfile(eventsFolder, pattern));
    if isempty(fileList)
        error('No CSV file found in %s for %s run %d', eventsFolder, subject, run);
    end
    filepath = fullfile(fileList(1).folder, fileList(1).name);

    % --- Read & check columns ---
    T = readtable(filepath, 'TextType','string', 'Delimiter',',');
    need = ["cue_start","trial_start"];
    for c = need
        if ~ismember(c, T.Properties.VariableNames)
            error('Column "%s" not found in %s', c, filepath);
        end
    end
    hasCueEnd = ismember("cue_end", T.Properties.VariableNames);
    
    % --- Select rows for this run ---
    if ismember("run", T.Properties.VariableNames)
        rows_run = (T.run == run);
    else
        rows_run = true(height(T),1);
    end
    if ~any(rows_run)
        error('Run %d not found in %s', run, filepath);
    end

    % --- Trigger = first trial_start within this run ---
    firstIdxInRun = find(rows_run, 1, 'first');
    trigger_time  = T.trial_start(firstIdxInRun);

    % --- Check if trial exists globally ---
    idx_trial_all = find(T.trial_idx == trial_id, 1);
    if isempty(idx_trial_all)
        error('Trial %d not found in %s!', trial_id, filepath);
    end
    trial_info = T(find(T.trial_idx == trial_id, 1), :);

    % --- Check whether this trial belongs to the requested run ---
    in_requested_run = trial_info.run == run;
   
    % --- Extract Names ---
    parseStim = @(s) string(regexp(s, "[^,'\[\] ]+", "match")); % transform string of list into array of strings
    row_idx    = find(T.trial_idx == trial_id, 1);  % use found row, not trial_id as index
    trial_stim = parseStim(T.stimuli(row_idx));
    cond_name_cue = map_stim_to_condition(trial_stim, invert_flag);
    stim_name_cue = shorten_stim_names(trial_stim);
    trial_out = T.outcome(row_idx);
    cond_name_out = map_out_to_condition(trial_out);


    % --- Compute onsets/durations --- 
    if strcmp(timepoint, 'cue')
        if in_requested_run
            onset = double(trial_info.cue_start - trigger_time);
            if hasCueEnd
                duration = double(trial_info.cue_end - trial_info.cue_start);
            else
                error('For trial %d no cue end exists in file %s', trial_id, filepath);
            end
            if isnan(onset) || isnan(duration)
                error('For trial %d there is no correct onset(%d), or duration(%d) in file %s', trial_id, onset, duration, filepath);
            end
        else
            onset = [];
            duration=[];
        end
        cond_name = cond_name_cue;
        stim_name = stim_name_cue;
    
    elseif strcmp(timepoint, 'outcome')
        if in_requested_run
            onset = double(trial_info.outcome_start - trigger_time);
            duration = double(trial_info.outcome_end - trial_info.outcome_start);
            if isnan(onset) || isnan(duration)
                error('For trial %d there is no correct onset(%d), or duration(%d) in file %s', trial_id, onset, duration, filepath);
            end
        else
            onset = [];
            duration=[];
        end
        cond_name = cond_name_out;
        stim_name = [];
    end
end

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
function finalString = shorten_stim_names(trial_stim)
    shortStim = strings(size(trial_stim));
    % first reduce each individual stimulus 
    for i = 1:numel(trial_stim)
        parts = split(trial_stim(i), "_");
        if numel(parts) == 2
            % animal_camel -> camel
            shortStim(i) = parts(2);
        elseif numel(parts) == 3
            % plant_ivy_purple -> ivy
            shortStim(i) = parts(2);
        else
            shortStim(i) = parts(end);
        end
    end
    % combine shortStim into one string
    if numel(shortStim) == 1
        finalString = shortStim;
    elseif numel(shortStim) == 2
        finalString = shortStim(2);
    elseif numel(shortStim) >= 3
        finalString = join(shortStim(2:end), "-");
    end
end