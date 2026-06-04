function [onsets, durations] = get_onsets_except(subject, run, bidsFolder, cond_name, trial_id, invert_flag)
%  
% Get spm parameters for a specific trial from bids/events file. 
% Usage:
%   [name, onset, duration] = get_onsets_except(sub, run, bidsFolder, cond_name, trial_id);
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
    
    % --- remove TOI trial if requested ---
    T_mint = T;
    if ~isempty(trial_id)
        T_mint = T(T.trial_idx ~= trial_id, :);
    end

    % --- subset to current run ---
    if ismember("run", T_mint.Properties.VariableNames)
        rows_run = T_mint.run == run;
        if ~any(rows_run)
            error('Run %d not found in %s', run, filepath);
        end
        T_run = T_mint(rows_run, :);
    else
        error('No run column in table');
    end

    % --- trigger = first trial_start within this run ---
    % Use T_mint (not T) because firstIdxInRun is an index into T_mint;
    % T has one extra row (the removed TOI), so using T here gives the wrong row.
    firstIdxInRun = find(rows_run, 1, 'first');
    trigger_time  = T_mint.trial_start(firstIdxInRun);
   
    % --- add condition column for cue and outcome --- 
    parseStim = @(s) string(regexp(s, "[^,'\[\] ]+", "match")); % transform string of list into array of strings
    nTrials = height(T_run);
    cond_name_cue = strings(nTrials,1);
    cond_name_out = strings(nTrials,1);
    for i=1:nTrials
        trial_stim = parseStim(T_run.stimuli(i));
        cond_name_cue(i) = map_stim_to_condition(trial_stim, invert_flag);
        trial_out = T_run.outcome(i);
        cond_name_out(i) = map_out_to_condition(trial_out);
    end
    T_run.cond_name_cue = cond_name_cue;
    T_run.cond_name_out = cond_name_out;

    % --- compute onsets/durations --- 
    if contains(cond_name, "reward")
        rows_cond = T_run.cond_name_out == cond_name;
        onsets = double(T_run.outcome_start(rows_cond) - trigger_time);
        durations = double(T_run.outcome_end(rows_cond) - T_run.outcome_start(rows_cond));
        if any(isnan(onsets)) || any(isnan(durations)) || isempty(onsets) || isempty(durations)
            error('For trial %d there is no correct onset or duration in file %s', trial_id, filepath);
        end
    else
        rows_cond = T_run.cond_name_cue == cond_name;
        onsets = double(T_run.cue_start(rows_cond) - trigger_time);
        if hasCueEnd
            durations = double(T_run.cue_end(rows_cond) - T_run.cue_start(rows_cond));
        else
            error('For trial %d no cue end exists in file %s', trial_id, filepath);
        end
        if any(isnan(onsets)) || any(isnan(durations)) || isempty(onsets) || isempty(durations)
            error('For trial %d there is no correct onset or duration in file %s', trial_id, filepath);
        end
    end
end