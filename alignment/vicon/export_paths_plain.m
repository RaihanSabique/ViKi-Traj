% export_paths_plain.m
% Export each <pid>_reaching_aligned.mat (a MATLAB table, which Python cannot read) to
% <pid>_reaching_paths_plain.mat (-v7) in the same folder: a plain struct array 'paths', one element
% per trial, with the game timing (trial and movement window), the game-object trajectory, the
% VICON crop and the review status. No marker data.
%
%   matlab -batch "export_paths_plain"

thisDir = fileparts(mfilename('fullpath'));
alignDir = fullfile(thisDir, '..', '..', 'outputs', 'game_vicon_aligned', 'reaching');
files = dir(fullfile(alignDir, 'XXXXXXX*_reaching_aligned.mat'));
for i = 1:numel(files)
    S = load(fullfile(alignDir, files(i).name), 'outputTable');
    T = S.outputTable;
    paths = struct([]);
    for r = 1:height(T)
        paths(r).row = r;
        paths(r).pid = T.pid{r};
        paths(r).visit = T.visit(r);
        paths(r).group_no = T.group_no(r);
        paths(r).trial_no = double(T.trial_no(r));
        paths(r).trial_hand = T.trial_hand{r};
        paths(r).trial_start_time = T.trial_start_time{r};
        paths(r).trial_end_time = T.trial_end_time{r};
        paths(r).relative_start_sec = T.relative_start_sec(r);
        paths(r).move_start_time = T.move_start_time{r};
        paths(r).move_end_time = T.move_end_time{r};
        paths(r).move_start_sec = T.move_start_sec(r);
        paths(r).move_duration_s = T.move_duration_s(r);
        paths(r).RMov_path = double(T.RMov_path{r});
        paths(r).RMov_time_step_sec = double(T.RMov_time_step_sec{r});
        paths(r).vicon_trial = T.vicon_trial{r};
        paths(r).vicon_start_frame = T.vicon_start_frame(r);
        paths(r).total_frames = T.total_frames(r);
        paths(r).framerate = T.framerate(r);
        paths(r).vicon_offset_sec = T.vicon_offset_sec(r);
        paths(r).vicon_coverage = T.vicon_coverage(r);
        paths(r).align_source = T.align_source{r};
        paths(r).review_status = T.review_status{r};
    end
    out = fullfile(alignDir, strrep(files(i).name, '_reaching_aligned.mat', '_reaching_paths_plain.mat'));
    save(out, 'paths', '-v7');
    fprintf('%s: %d paths\n', out, height(T));
end
