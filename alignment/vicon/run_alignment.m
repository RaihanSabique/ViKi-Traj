% run_alignment.m
% Align the trials of the reaching game with VICON (align_participant) for one, several or all
% participants. Writes <pid>_reaching_aligned.mat (table 'outputTable', one row per trial),
% vicon_alignment_diagnostics.csv (one row per logged trial) and vicon_alignment_summary.csv
% (one row per participant) to outputs/game_vicon_aligned/reaching.
%
%   matlab -batch "participants={'XXXXXXX115'}; run_alignment"
%   matlab -batch "participants='all'; run_alignment"
%
% outDir and manualOnlyParticipants can be set the same way.

thisDir = fileparts(mfilename('fullpath'));
repoDir = fullfile(thisDir, '..', '..');
gameDir = fullfile(repoDir, 'data', 'game_data');
viconDir = fullfile(repoDir, 'data', 'vicon_data');
manualXlsx = fullfile(repoDir, 'data', 'Vicon Label Marker Tracker Update.xlsx');
if ~exist('outDir', 'var')
    outDir = fullfile(repoDir, 'outputs', 'game_vicon_aligned', 'reaching');
end
if ~exist(outDir, 'dir'), mkdir(outDir); end

if ~exist('participants', 'var')
    participants = {'XXXXXXX115'};
end
if ischar(participants) && strcmpi(participants, 'all')
    vf = dir(fullfile(viconDir, 'XXXXXXX*_VICON.mat'));
    participants = regexp({vf.name}, '^(XXXXXXX\d+)_VICON\.mat$', 'tokens', 'once');
    participants = [participants{:}];
end
participants = cellstr(participants);

% Participants (UCP) whose finger speed does not follow the game object closely enough for the
% motion-based estimates: only their manual starts are used.
if ~exist('manualOnlyParticipants', 'var')
    manualOnlyParticipants = {'XXXXXXX202', 'XXXXXXX207', 'XXXXXXX216', 'XXXXXXX218'};
end

summary = table();
allDiag = table();
for i = 1:numel(participants)
    pid = participants{i};
    gameFile = fullfile(gameDir, [pid '.mat']);
    viconFile = fullfile(viconDir, [pid '_VICON.mat']);
    row = table(string(pid), "", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NaN, NaN, ...
        'VariableNames', {'pid', 'status', 'paths', 'need_review', 'no_vicon', 'manual', 'xcorr', 'trial_consensus', 'spike', ...
        'unverified', 'truncated', 'manual_rejected', 'median_abs_manual_minus_xcorr_sec', 'max_abs_manual_minus_xcorr_sec'});
    if ~isfile(gameFile) || ~isfile(viconFile)
        row.status = "missing game or vicon file";
        fprintf('%s: skipped (%s)\n', pid, row.status);
        summary = [summary; row]; %#ok<AGROW>
        continue;
    end
    try
        manualOnly = ismember(pid, manualOnlyParticipants);
        [outputTable, diagTable] = align_participant(pid, gameFile, viconFile, manualXlsx, 'ManualOnly', manualOnly);
    catch ME
        row.status = "error: " + string(ME.message);
        fprintf('%s: FAILED %s\n', pid, ME.message);
        summary = [summary; row]; %#ok<AGROW>
        continue;
    end
    save(fullfile(outDir, [pid '_reaching_aligned.mat']), 'outputTable', '-v7');

    diagTable = addvars(diagTable, repmat(string(pid), height(diagTable), 1), 'Before', 1, 'NewVariableNames', 'pid');
    allDiag = [allDiag; diagTable]; %#ok<AGROW>
    src = string(outputTable.align_source);
    row.status = "ok";
    row.paths = height(outputTable);
    row.no_vicon = nnz(outputTable.total_frames == 0);
    row.need_review = nnz(strcmp(outputTable.review_status, 'Need Review'));
    row.manual = nnz(src == "manual");
    row.xcorr = nnz(src == "xcorr");
    row.trial_consensus = nnz(src == "trial_consensus" | src == "manual_trial_median");
    row.spike = nnz(src == "spike");
    sheetOnly = manualOnly & ismember(src, ["manual", "manual_trial_median"]);
    row.unverified = nnz(outputTable.align_support <= 1 & ~sheetOnly & outputTable.total_frames > 0);
    row.truncated = nnz(outputTable.vicon_truncated);
    row.manual_rejected = nnz(~isnan(outputTable.manual_start_sec) & src ~= "manual");
    d = abs(diagTable.manual_offset_sec - diagTable.xcorr_offset_sec);
    d = d(diagTable.xcorr_corr >= 0.7);     % voting cross-correlations only (MinCorr of align_participant)
    row.median_abs_manual_minus_xcorr_sec = median(d, 'omitnan');
    if any(~isnan(d))
        row.max_abs_manual_minus_xcorr_sec = max(d, [], 'omitnan');
    end
    summary = [summary; row]; %#ok<AGROW>

    fprintf('\n=== %s: %d paths, %d without VICON data (manual %d, xcorr %d, consensus %d, spike %d, unverified %d, truncated %d) ===\n', ...
        pid, height(outputTable), row.no_vicon, row.manual, row.xcorr, row.trial_consensus, row.spike, row.unverified, row.truncated);
    disp(diagTable(:, {'study_day', 'trial_no', 'group_no', 'relative_start_sec', 'align_source', 'align_support', ...
        'vicon_offset_sec', 'manual_offset_sec', 'xcorr_offset_sec', 'xcorr_corr', 'spike_offset_sec', 'total_frames', 'vicon_truncated', 'kept'}));
end

writetable(summary, fullfile(outDir, 'vicon_alignment_summary.csv'));
writetable(allDiag, fullfile(outDir, 'vicon_alignment_diagnostics.csv'));
disp(summary);
