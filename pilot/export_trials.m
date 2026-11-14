function export_trials(pid)
% export_trials  Write the provided aligned trial table of the pilot participant as a plain struct array.
%
%   matlab -batch "cd pilot; export_trials('XXXXXXX115')"
%
%   Reads data/<pid>/<pid>_reaching_aligned_provided.mat (a MATLAB table, which scipy cannot read) and
%   writes data/<pid>/<pid>_reaching_plain.mat with the struct array 'trials' (-v7), one element per
%   trial, holding the columns that align_trials.py and evaluate_pilot.py read.
%
%   The input is the earlier aligned trial table that came with the data, made before and outside
%   alignment/vicon. Its VICON crops of the second visit are 0.8-1.3 s out of step with the game
%   trajectory, which the pilot analysis exposed; run_alignment.m corrects this offset.

repoDir = fullfile(fileparts(mfilename('fullpath')), '..');
src = fullfile(repoDir, 'data', pid, [pid '_reaching_aligned_provided.mat']);
dst = fullfile(repoDir, 'data', pid, [pid '_reaching_plain.mat']);
S = load(src);
fn = fieldnames(S);
T = S.(fn{1});
cols = {'pid', 'study_day', 'trial_hand', 'trial_no', 'trial_start_time', 'trial_end_time', ...
        'road_path', 'target_pos', 'rocket_pos', 'RT_lift_pos', 'RT_lift_time', 'RMov_path', ...
        'RMov_time', 'RMov_time_step_ms', 'time_diff_s', 'RMov_dist_px', 'relative_start_sec', ...
        'group_no', 'position_data_interpolated', 'time', 'framerate', 'labels', ...
        'total_time_vicon_s', 'total_frames', 'age'};
trials = struct();
for r = 1:height(T)
    for c = 1:numel(cols)
        x = T{r, cols{c}};
        if iscell(x) && numel(x) == 1, x = x{1}; end
        if ischar(x) && size(x, 1) > 1, x = cellstr(x); end    % timestamps stored as an N x 24 char array
        if isstring(x), x = cellstr(x); end
        trials(r).(cols{c}) = x;
    end
end
save(dst, 'trials', '-v7');
fprintf('%s: %d trials written to %s\n', pid, numel(trials), dst);
end
