% export_vicon_trial_frames.m
% Frame count of every VICON recording of the reaching game, written to
% outputs/game_vicon_aligned/reaching/vicon_trial_frames.csv (file, trial, game, n_frames).
% analysis/make_tables.py reads it for the number and total length of the recordings.
%
%   matlab -batch "export_vicon_trial_frames"

thisDir = fileparts(mfilename('fullpath'));
root = fullfile(thisDir, '..', '..');
f = dir(fullfile(root, 'data', 'vicon_data', '*.mat'));
out = {};
for i = 1:numel(f)
    S = load(fullfile(f(i).folder, f(i).name));
    fn = fieldnames(S);
    v = S.(fn{1});
    for k = 1:numel(v)
        if startsWith(string(v(k).game), 'Reaching', 'IgnoreCase', true)
            out(end+1, :) = {f(i).name, char(string(v(k).trial)), char(string(v(k).game)), numel(v(k).time)}; %#ok<SAGROW>
        end
    end
end
writetable(cell2table(out, 'VariableNames', {'file', 'trial', 'game', 'n_frames'}), ...
    fullfile(root, 'outputs', 'game_vicon_aligned', 'reaching', 'vicon_trial_frames.csv'));
