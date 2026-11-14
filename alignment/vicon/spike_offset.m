function [offsetSec, info] = spike_offset(viconTime, positionData, labels, groupTbl)
% spike_offset VICON-to-game clock offset from the finger lift at the end of trial 1.
%
%   [offsetSec, info] = spike_offset(viconTime, positionData, labels, groupTbl)
%
%   The finger marker of the hand of trial 1 (RFIN, or LFIN for the left hand) is lifted when
%   trial 1 ends. The largest frame-to-frame change of its Y coordinate between the end of
%   trial 1 and the start of trial 2 (or the end of trial 1 + 8 s) is taken as this lift:
%       vicon_time = game_time_relative_to_trial1_start + offsetSec
%   The finger lifts after trial_end_time, so the offset is biased by about +1 s; it is used
%   only when a recording has no other estimate. NaN when the marker has no data in the window.
%
%   groupTbl  game rows of one VICON recording, sorted by trial_start_time, with variables
%             relative_start_sec, trial_end_sec (s from the start of trial 1) and trial_hand.
%   info      marker, VICON time and dY of the lift, and the search window.

offsetSec = NaN;
info = struct('channel', '', 'spikeTime', NaN, 'searchWindow', [NaN NaN], 'derivValue', NaN);

hand = lower(char(string(groupTbl.trial_hand{1})));
if contains(hand, 'left')
    chanName = 'LFIN';
else
    chanName = 'RFIN';
end
chanIdx = find(strcmpi(labels, chanName), 1);
if isempty(chanIdx)
    return;
end
info.channel = chanName;

y = squeeze(positionData(2, chanIdx, :))';
dY = [0 diff(y)];

trial1End = groupTbl.trial_end_sec(1);
if height(groupTbl) >= 2 && groupTbl.relative_start_sec(2) > trial1End
    searchEnd = groupTbl.relative_start_sec(2);
else
    searchEnd = trial1End + 8;
end
info.searchWindow = [trial1End searchEnd];

idxWin = find(viconTime(:)' >= trial1End & viconTime(:)' <= searchEnd);
if isempty(idxWin) || all(isnan(dY(idxWin)))
    return;
end
[~, loc] = max(abs(dY(idxWin)));       % first occurrence of the largest spike
k = idxWin(loc);
info.spikeTime = viconTime(k);
info.derivValue = dY(k);
offsetSec = viconTime(k) - trial1End;
end
