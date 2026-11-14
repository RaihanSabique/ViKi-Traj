% summarize_aligned.m
% Per-trial and per-participant summary of the aligned files: round of the visit (vicon_round),
% alignment source, VICON coverage and missing marker data (NaN frames). Writes aligned_paths.csv
% (one row per trial in the order of outputTable, read by build_benchmark_manifest.py and
% trajectory_check.py) and aligned_participants.csv next to the aligned files.
%
%   matlab -batch "summarize_aligned"

thisDir = fileparts(mfilename('fullpath'));
alignDir = fullfile(thisDir, '..', '..', 'outputs', 'game_vicon_aligned', 'reaching');
files = dir(fullfile(alignDir, 'XXXXXXX*_reaching_aligned.mat'));

paths = table();
for i = 1:numel(files)
    S = load(fullfile(alignDir, files(i).name), 'outputTable');
    T = S.outputTable;
    pid = string(erase(files(i).name, '_reaching_aligned.mat'));
    for r = 1:height(T)
        labels = T.labels{r};
        P = T.position_data_interpolated{r};
        nFr = size(P, 3);
        hand = string(T.trial_hand{r});
        finger = "RFIN";
        if contains(lower(hand), "left"), finger = "LFIN"; end
        c = find(strcmpi(labels, finger), 1);
        if nFr > 0
            nanFrame = squeeze(any(isnan(P), 1));                  % markers x frames
            fingerNaN = 100 * mean(nanFrame(c, :));
            anyNaN = 100 * mean(any(nanFrame, 1));
            markerNaN = 100 * mean(nanFrame, 2);
            emptyMarkers = strjoin(string(labels(markerNaN == 100)), ' ');
            worst = max(markerNaN);
        else
            fingerNaN = NaN; anyNaN = NaN; worst = NaN; emptyMarkers = "";
        end
        tok = regexp(string(T.vicon_trial{r}), 'Reaching (\d+)', 'tokens', 'once');   % round n of 'Reaching n'
        vt = NaN;
        if ~isempty(tok), vt = str2double(tok{1}); end
        paths = [paths; table(pid, T.visit(r), T.group_no(r), vt, double(T.trial_no(r)), hand, ...
            string(T.align_source{r}), T.align_support(r), T.move_duration_s(r), nFr, T.vicon_coverage(r), ...
            fingerNaN, anyNaN, worst, emptyMarkers, string(T.review_status{r}), ...
            'VariableNames', {'pid', 'visit', 'group_no', 'vicon_round', 'path', 'hand', 'align_source', 'align_support', ...
            'move_duration_s', 'vicon_frames', 'vicon_coverage', 'finger_nan_pct', 'any_marker_nan_pct', ...
            'worst_marker_nan_pct', 'fully_missing_markers', 'review_status'})]; %#ok<AGROW>
    end
end
writetable(paths, fullfile(alignDir, 'aligned_paths.csv'));

% Per participant
[G, pid] = findgroups(paths.pid);
part = table(pid, ...
    splitapply(@numel, paths.path, G), ...
    splitapply(@(s) nnz(s == "Need Review"), paths.review_status, G), ...
    splitapply(@(c) nnz(c >= 0.999), paths.vicon_coverage, G), ...
    splitapply(@(c) nnz(c > 0 & c < 0.999), paths.vicon_coverage, G), ...
    splitapply(@(c) nnz(c == 0), paths.vicon_coverage, G), ...
    splitapply(@(s) nnz(s == "manual" | s == "manual_trial_median"), paths.align_source, G), ...
    splitapply(@(s) nnz(s == "xcorr" | s == "trial_consensus"), paths.align_source, G), ...
    splitapply(@(x) median(x, 'omitnan'), paths.finger_nan_pct, G), ...
    splitapply(@(x) max(x, [], 'omitnan'), paths.finger_nan_pct, G), ...
    splitapply(@(x) nnz(x > 10), paths.finger_nan_pct, G), ...
    'VariableNames', {'pid', 'paths', 'need_review', 'full_vicon', 'partial_vicon', 'no_vicon', 'from_sheet', 'from_motion', ...
    'median_finger_nan_pct', 'max_finger_nan_pct', 'paths_finger_nan_gt10pct'});
writetable(part, fullfile(alignDir, 'aligned_participants.csv'));
disp(part);
