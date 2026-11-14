% export_markers.m
% Export the VICON markers of every aligned trial from <pid>_reaching_aligned.mat (a MATLAB table,
% which Python cannot read) to outputs/benchmark/markers/<pid>_reaching_markers.mat (-v7): a plain
% struct array 'paths', one element per trial in the order of outputTable (field row), with
%   X            3 x markers x N VICON positions (mm) over the movement window
%   time         1 x N game time (s from trial_start_time of trial 1 of the group)
%   labels       marker labels; framerate, vicon_coverage, vicon_start_frame
%   trial_start_time, relative_start_sec, move_start_time   to place 'time' on the game clock:
%                absolute game time = trial_start_time - relative_start_sec + time
%
%   matlab -batch "export_markers"

thisDir = fileparts(mfilename('fullpath'));
root = fullfile(thisDir, '..', '..');
alignDir = fullfile(root, 'outputs', 'game_vicon_aligned', 'reaching');
outDir = fullfile(root, 'outputs', 'benchmark', 'markers');
if ~exist(outDir, 'dir'), mkdir(outDir); end
files = dir(fullfile(alignDir, 'XXXXXXX*_reaching_aligned.mat'));
for i = 1:numel(files)
    S = load(fullfile(alignDir, files(i).name), 'outputTable');
    T = S.outputTable;
    paths = struct([]);
    for r = 1:height(T)
        paths(r).row = r;
        paths(r).pid = T.pid{r};
        paths(r).visit = T.visit(r);
        paths(r).trial_no = double(T.trial_no(r));
        paths(r).trial_hand = T.trial_hand{r};
        paths(r).trial_start_time = T.trial_start_time{r};
        paths(r).relative_start_sec = T.relative_start_sec(r);
        paths(r).move_start_time = T.move_start_time{r};
        paths(r).X = double(T.position_data_interpolated{r});
        paths(r).time = double(T.time{r});
        paths(r).labels = T.labels{r};
        paths(r).framerate = T.framerate(r);
        paths(r).vicon_coverage = T.vicon_coverage(r);
        paths(r).vicon_start_frame = T.vicon_start_frame(r);
    end
    out = fullfile(outDir, strrep(files(i).name, '_reaching_aligned.mat', '_reaching_markers.mat'));
    save(out, 'paths', '-v7');
    fprintf('%s: %d paths\n', out, height(T));
end
