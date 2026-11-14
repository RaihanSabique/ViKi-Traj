function [outputTable, diagTable] = align_participant(participantId, gameFile, viconFile, manualXlsx, varargin)
% align_participant Align the trials of the reaching game of one participant with VICON.
%
%   [outputTable, diagTable] = align_participant(pid, gameFile, viconFile, manualXlsx, ...)
%
%   A trial is one row of the game table (a path in the game data, trial_no 1-4). The 4 trials
%   of a VICON recording ('Reaching n', one group) share one VICON-minus-game clock offset. Each
%   trial gives two estimates: its manual start in the 'Frame Times' sheet (read_manual_starts)
%   and the cross-correlation of finger-marker and game-object speed (xcorr_offset). All
%   estimates of a recording vote; the offset shared within 'TolSec' by the most estimates is the
%   consensus. align_source of a trial, in order of preference:
%     'manual'           its manual start agrees with the consensus
%     'xcorr'            its cross-correlation offset agrees with the consensus
%     'trial_consensus'  the consensus offset
%     'spike'            no estimate in the recording: finger lift after trial 1 (spike_offset)
%   align_support counts the estimates behind the consensus (1 = nothing agreed: unverified).
%   With 'ManualOnly', a recording with manual starts uses only these: a trial takes its own
%   manual start ('manual'), else the median manual offset of the recording ('manual_trial_median').
%
%   The VICON crop covers the movement window of the trial, from the first to the last RMov_time
%   sample (move_start_time, move_end_time). Aborted attempts (an incomplete group restarted later
%   in the same visit) are dropped. Every other trial is kept; a trial without VICON data (no
%   recording, no offset, or outside the recording) has empty VICON arrays and vicon_coverage = 0.
%
%   Name-value options:
%     'TolSec'      agreement tolerance between estimates (s), default 0.1
%     'MinCorr'     minimum speed correlation of a voting cross-correlation estimate, default 0.7
%     'ManualOnly'  use only the manual starts, default false
%
%   outputTable  one row per trial: the game columns, the VICON crop (position_data_interpolated,
%                time, framerate, labels, total_time_vicon_s, total_frames) and the alignment
%                columns, each described in Properties.VariableDescriptions. 'time' is game time
%                (s) from the start of trial 1 of the group.
%   diagTable    alignment diagnostics of every logged trial (kept = false for aborted attempts)

p = inputParser;
p.addParameter('TolSec', 0.1, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('MinCorr', 0.7, @(x) isnumeric(x) && isscalar(x));
p.addParameter('ManualOnly', false, @(x) islogical(x) || isnumeric(x));
p.parse(varargin{:});
manualOnly = logical(p.Results.ManualOnly);
tol = p.Results.TolSec;
minCorr = p.Results.MinCorr;

%% Trials of the game, grouped by VICON recording
G = load(gameFile, 'Reaching');
gs = G.Reaching;
gameProps = [];
if isfield(gs, 'Properties')
    gameProps = gs.Properties;
    gs = rmfield(gs, 'Properties');
end
game = struct2table(gs);
gameVars = game.Properties.VariableNames;

startDT = parseTime(game.trial_start_time);
endDT = parseTime(game.trial_end_time);
[~, order] = sort(startDT);
game = game(order, :); startDT = startDT(order); endDT = endDT(order);

visitOfRow = dayToVisit(game.study_day);
tn = double(game.trial_no);
groupOfRow = zeros(height(game), 1);
g = 0;
for i = 1:height(game)
    if i == 1 || visitOfRow(i) ~= visitOfRow(i-1) || tn(i) <= tn(i-1)
        g = g + 1;
    end
    groupOfRow(i) = g;
end
nGroups = g;
groupVisit = arrayfun(@(k) visitOfRow(find(groupOfRow == k, 1)), 1:nGroups);

%% VICON recordings of the reaching game per visit
V = load(viconFile);
fn = fieldnames(V);
vicon = V.(fn{1});
vGame = string({vicon.game});
vTrial = string({vicon.trial});
vRec = find(startsWith(vGame, 'Reaching', 'IgnoreCase', true));     % game 'Reaching 1' to 'Reaching 3'
vVisit = nan(size(vRec));
vGameNo = nan(size(vRec));
for k = 1:numel(vRec)
    tok = regexp(vTrial(vRec(k)), 'V(\d)', 'tokens', 'once');
    if ~isempty(tok), vVisit(k) = str2double(tok{1}); end
    tok = regexp(vGame(vRec(k)), '(\d+)\s*$', 'tokens', 'once');
    if ~isempty(tok), vGameNo(k) = str2double(tok{1}); end
end
if any(isnan(vVisit))
    % No visit in the recording name: visit 2 starts at the second calibration recording
    calRows = find(contains(vGame, 'Calibration', 'IgnoreCase', true));
    for k = find(isnan(vVisit))
        vVisit(k) = 1 + (numel(calRows) >= 2 && vRec(k) > calRows(2));
    end
end

%% Map each group to a VICON recording (chronological within a visit)
groupVicon = nan(1, nGroups);       % index into the vicon struct array
groupManualTrial = nan(1, nGroups); % 'Trial n' column of the manual sheet
for v = unique(groupVisit)
    gk = find(groupVisit == v);
    vk = find(vVisit == v);
    if numel(gk) == numel(vk)
        groupVicon(gk) = vRec(vk);
        mapK = vk;
    else
        warning('reaching:map', '%s V%d: %d game groups vs %d VICON trials; mapping by game number.', ...
            participantId, v, numel(gk), numel(vk));
        mapK = nan(size(gk));
        for j = 1:numel(gk)
            hit = vk(vGameNo(vk) == j);
            if isscalar(hit)
                mapK(j) = hit;
                groupVicon(gk(j)) = vRec(hit);
            end
        end
    end
    % The manual 'Trial n' annotates VICON 'Reaching n'; for a repeated (restarted)
    % recording it belongs to the last recording of that name.
    for j = 1:numel(gk)
        if isnan(mapK(j)), continue; end
        n = vGameNo(mapK(j));
        if ~isnan(n) && n <= 3 && mapK(j) == find(vVisit == v & vGameNo == n, 1, 'last')
            groupManualTrial(gk(j)) = n;
        end
    end
end

manual = read_manual_starts(manualXlsx, participantId);

%% Aborted attempts: an incomplete group restarted later in the same visit
aborted = false(1, nGroups);
for g = 1:nGroups
    aborted(g) = nnz(groupOfRow == g) < 4 && any(groupVisit(g+1:end) == groupVisit(g));
end
groupNoOut = nan(1, nGroups);
groupNoOut(~aborted) = 1:nnz(~aborted);

% Marker labels and frame rate for trials without VICON data
labels0 = vicon(vRec(1)).labels;
rate0 = double(vicon(vRec(1)).framerate);

%% Align every trial
nRows = height(game);
relStart = nan(nRows, 1);
durSec = seconds(endDT - startDT);
% Movement window (s from trial_start_time): first and last RMov_time sample
moveLag = zeros(nRows, 1); moveEnd = durSec;
moveStartStr = strings(nRows, 1); moveEndStr = strings(nRows, 1);
for r = 1:nRows
    ts = double(game.RMov_time_step_sec{r});
    rm = string(game.RMov_time{r});
    if ~isempty(ts) && ~isempty(rm)
        [moveLag(r), i0] = min(ts); [moveEnd(r), i1] = max(ts);
        moveStartStr(r) = strtrim(rm(i0)); moveEndStr(r) = strtrim(rm(i1));
    else
        moveStartStr(r) = string(game.trial_start_time{r}); moveEndStr(r) = string(game.trial_end_time{r});
    end
end
moveDur = moveEnd - moveLag;
pos = repmat({zeros(3, numel(labels0), 0)}, nRows, 1); tvec = repmat({zeros(1, 0)}, nRows, 1);
labelsCol = repmat({labels0}, nRows, 1);
fr = repmat(rate0, nRows, 1); totTime = zeros(nRows, 1); totFrames = zeros(nRows, 1);
viconTrialCol = strings(nRows, 1); sourceCol = strings(nRows, 1);
startFrame = nan(nRows, 1); offsetCol = nan(nRows, 1); supportCol = zeros(nRows, 1);
manualCol = nan(nRows, 1); manOffCol = nan(nRows, 1); manMarkCol = strings(nRows, 1); xcPathCol = nan(nRows, 1); xcPathCorrCol = nan(nRows, 1);
consCol = nan(nRows, 1); xcTrialCol = nan(nRows, 1); xcTrialCorrCol = nan(nRows, 1); spikeCol = nan(nRows, 1);
truncCol = false(nRows, 1); coverageCol = zeros(nRows, 1);
keep = true(nRows, 1);

for g = 1:nGroups
    rows = find(groupOfRow == g);
    relStart(rows) = seconds(startDT(rows) - startDT(rows(1)));
    if aborted(g)
        warning('reaching:align', '%s group %d (V%d): aborted attempt (%d path) restarted later; dropped.', ...
            participantId, g, groupVisit(g), numel(rows));
        keep(rows) = false;
        continue;
    end
    if isnan(groupVicon(g))
        warning('reaching:map', '%s group %d (V%d): no VICON trial; %d paths kept without VICON data.', ...
            participantId, g, groupVisit(g), numel(rows));
        sourceCol(rows) = "no_vicon_trial";
        truncCol(rows) = true;
        continue;
    end
    vr = vicon(groupVicon(g));
    t = double(vr.time(:));
    rate = double(vr.framerate);
    P = vr.position_data_interpolated;
    labels = vr.labels;

    % Manual estimates (VICON minus game time). The sheet marks either the trial start
    % (trial_start_time) or the movement start, not consistently, so each entry gives two
    % candidates; the one that agrees with its recording is used.
    manSec = nan(numel(rows), 1);
    if ~isnan(groupManualTrial(g))
        for j = 1:numel(rows)
            pathNo = tn(rows(j));
            if pathNo >= 1 && pathNo <= 4
                manSec(j) = manual(groupVisit(g), groupManualTrial(g), pathNo);
            end
        end
    end
    nr = numel(rows);
    manCand = [manSec - relStart(rows), manSec - (relStart(rows) + moveLag(rows))];  % [trial start, movement start]
    useManualOnly = manualOnly && any(~isnan(manSec));

    % Motion-based estimates (skipped when only the manual starts are used)
    gt = table(relStart(rows), relStart(rows) + durSec(rows), game.trial_hand(rows), ...
        game.RMov_path(rows), game.RMov_time_step_sec(rows), 'VariableNames', ...
        {'relative_start_sec', 'trial_end_sec', 'trial_hand', 'RMov_path', 'RMov_time_step_sec'});
    if useManualOnly
        xcTrial = NaN; spikeOff = NaN;
        xcInfo = struct('corr', NaN, 'pathOffset', nan(numel(rows), 1), 'pathCorr', nan(numel(rows), 1));
    else
        [xcTrial, xcInfo] = xcorr_offset(t, P, labels, gt);
        spikeOff = spike_offset(t, P, labels, gt);
    end
    xcOff = xcInfo.pathOffset;
    xcOff(~(xcInfo.pathCorr >= minCorr)) = NaN;
    if all(isnan(xcOff)) && xcInfo.corr >= minCorr
        xcOff(1) = xcTrial;              % no single trial matched; use all trials jointly
    end
    % Vote: the manual entry (both readings) and the cross-correlation of each trial count once each
    if useManualOnly
        [cons, support] = consensusOffset(manCand(:), zeros(2 * nr, 1), tol, [1:nr, 1:nr]');
    else
        [cons, support] = consensusOffset([manCand(:); xcOff], [zeros(2 * nr, 1); xcInfo.pathCorr], tol, ...
            [1:nr, 1:nr, nr + (1:nr)]');
    end
    [manOff, manMark] = pickManual(manCand, cons);
    if useManualOnly && support < 2
        cons = median(manCand(:, 1), 'omitnan');     % no two sheet entries agree: trial-start reading
        [manOff, manMark] = pickManual(manCand, cons);
    end
    if useManualOnly
        cons = median(manOff, 'omitnan');
    elseif manualOnly
        warning('reaching:align', '%s group %d: no manual start in this VICON trial; using motion-based offset %.3f s.', ...
            participantId, g, cons);
    end
    if support == 1 && ~useManualOnly
        warning('reaching:consensus', '%s group %d: manual and cross-correlation estimates disagree; offset %.3f s is unverified.', ...
            participantId, g, cons);
    end

    for j = 1:numel(rows)
        r = rows(j);
        viconTrialCol(r) = vTrial(groupVicon(g)) + " | " + vGame(groupVicon(g));
        labelsCol{r} = labels;
        fr(r) = rate;
        pos{r} = P(:, :, []);
        supportCol(r) = support;
        truncCol(r) = true;
        if useManualOnly && ~isnan(manOff(j))
            off = manOff(j); src = "manual";
        elseif useManualOnly
            off = cons; src = "manual_trial_median";
        elseif abs(manOff(j) - cons) <= tol
            off = manOff(j); src = "manual";
        elseif abs(xcOff(j) - cons) <= tol
            off = xcOff(j); src = "xcorr";
        elseif ~isnan(cons)
            off = cons; src = "trial_consensus";
        elseif ~isnan(spikeOff)
            off = spikeOff; src = "spike";
        else
            warning('reaching:align', '%s group %d path %d: no offset estimate; kept without VICON data.', participantId, g, tn(r));
            sourceCol(r) = "none";
            continue;
        end
        sourceCol(r) = src;
        offsetCol(r) = off;

        % VICON frames covering the movement window [move start, move end] in game time
        w0 = relStart(r) + moveLag(r); w1 = relStart(r) + moveEnd(r);
        nFrames = floor(moveDur(r) * rate + 1e-6) + 1;
        i0 = find(t >= w0 + off - 0.5 / rate, 1, 'first');
        i1 = find(t <= w1 + off + 0.5 / rate, 1, 'last');
        if isempty(i0) || isempty(i1) || i1 < i0
            warning('reaching:align', '%s group %d path %d: movement (%.2f-%.2f s VICON time) outside VICON trial (%.2f-%.2f s); kept without VICON data.', ...
                participantId, g, tn(r), w0 + off, w1 + off, t(1), t(end));
            continue;
        end
        i1 = min(i1, i0 + nFrames - 1);
        idx = i0:i1;

        pos{r} = P(:, :, idx);
        tvec{r} = (t(idx) - off)';
        totTime(r) = tvec{r}(end) - tvec{r}(1);
        totFrames(r) = numel(idx);
        startFrame(r) = i0 - 1;             % 0-based frame (= VICON time x framerate)
        coverageCol(r) = numel(idx) / nFrames;
        truncCol(r) = numel(idx) < nFrames || w0 + off < t(1) - 0.5 / rate;
    end
    manualCol(rows) = manSec; manOffCol(rows) = manOff; manMarkCol(rows) = manMark;
    xcPathCol(rows) = xcInfo.pathOffset; xcPathCorrCol(rows) = xcInfo.pathCorr;
    consCol(rows) = cons; xcTrialCol(rows) = xcTrial; xcTrialCorrCol(rows) = xcInfo.corr; spikeCol(rows) = spikeOff;
end

%% Output table
outputTable = game;
outputTable.relative_start_sec = relStart;
outputTable.group_no = groupNoOut(groupOfRow)';
outputTable.position_data_interpolated = pos;
outputTable.time = tvec;
outputTable.framerate = fr;
outputTable.labels = labelsCol;
outputTable.total_time_vicon_s = totTime;
outputTable.total_frames = totFrames;
outputTable.visit = visitOfRow;
outputTable.move_start_time = cellstr(moveStartStr);
outputTable.move_end_time = cellstr(moveEndStr);
outputTable.move_start_sec = relStart + moveLag;
outputTable.move_duration_s = moveDur;
outputTable.vicon_trial = cellstr(viconTrialCol);
outputTable.align_source = cellstr(sourceCol);
outputTable.align_support = supportCol;
outputTable.vicon_start_frame = startFrame;
outputTable.vicon_offset_sec = offsetCol;
outputTable.manual_start_sec = manualCol;
outputTable.manual_mark = cellstr(manMarkCol);
outputTable.xcorr_offset_sec = xcPathCol;
outputTable.xcorr_corr = xcPathCorrCol;
outputTable.vicon_truncated = truncCol;
outputTable.vicon_coverage = coverageCol;
[reviewStatus, reviewReason] = reviewPaths(outputTable, groupOfRow, manOffCol, manualOnly);
outputTable.review_status = reviewStatus;

desc = repmat({''}, 1, width(outputTable));
if ~isempty(gameProps)
    [tf, loc] = ismember(gameVars, gameProps.VariableNames);
    desc(tf) = gameProps.VariableDescriptions(loc(tf));
end
added = { ...
    'relative_start_sec', 'Path start (s) relative to path-1 start of its VICON trial (group)'; ...
    'group_no', 'Running index of the VICON Reaching trial (4 paths each) across both visits'; ...
    'position_data_interpolated', 'Aligned VICON marker positions of the movement window of this path (3 x markers x frames)'; ...
    'time', 'Game time (s, relative to path-1 start of the group) of each VICON frame'; ...
    'framerate', 'VICON frame rate (Hz)'; ...
    'labels', 'VICON marker labels'; ...
    'total_time_vicon_s', 'Duration of the cropped VICON segment (s)'; ...
    'total_frames', 'Number of VICON frames in the cropped segment'; ...
    'visit', 'Visit number (1 = Day-1, 2 = Day-2)'; ...
    'move_start_time', 'Game-object movement start: first RMov_time sample'; ...
    'move_end_time', 'Game-object movement end: last RMov_time sample'; ...
    'move_start_sec', 'Movement start (s) relative to path-1 start of its VICON trial (group); the VICON crop starts here'; ...
    'move_duration_s', 'Movement window duration (s): last minus first RMov_time sample'; ...
    'vicon_trial', 'Source VICON trial and game name'; ...
    'align_source', 'manual | xcorr | trial_consensus | spike | manual_trial_median (manual-only participants) | no_vicon_trial | none'; ...
    'align_support', 'Number of manual + cross-correlation estimates of the VICON trial agreeing with the offset (1 = unverified)'; ...
    'vicon_start_frame', 'First cropped VICON frame (0-based, = VICON time x framerate)'; ...
    'vicon_offset_sec', 'VICON time minus game time used for this path (s)'; ...
    'manual_start_sec', 'Manual VICON start of this path from the Frame Times sheet (s), NaN if not annotated'; ...
    'manual_mark', 'What the sheet entry marks, the reading agreeing with its VICON trial: path_start (trial_start_time) or move_start (first RMov_time)'; ...
    'xcorr_offset_sec', 'Offset from finger/game-object speed cross-correlation of this path (s)'; ...
    'xcorr_corr', 'Speed correlation at xcorr_offset_sec'; ...
    'vicon_truncated', 'True if the VICON trial ended (or started) inside the path, or has no VICON data'; ...
    'vicon_coverage', 'Fraction of the movement window covered by VICON frames (0 = no VICON data)'; ...
    'review_status', 'OK or Need Review (unverified offset, inconsistent manual start, VICON coverage < 90 %, or active-hand finger marker missing > 10 % of frames)'};
[~, loc] = ismember(added(:,1), outputTable.Properties.VariableNames);
desc(loc) = added(:,2);
outputTable.Properties.VariableDescriptions = desc;

diagTable = outputTable(:, {'study_day', 'trial_hand', 'trial_no', 'group_no', 'visit', 'vicon_trial', ...
    'relative_start_sec', 'move_start_sec', 'move_duration_s', 'align_source', 'align_support', 'vicon_offset_sec', 'manual_start_sec', 'manual_mark', ...
    'xcorr_offset_sec', 'xcorr_corr', 'vicon_start_frame', 'total_frames', 'vicon_truncated', 'vicon_coverage'});
diagTable.manual_offset_sec = manOffCol;
diagTable.consensus_offset_sec = consCol;
diagTable.xcorr_trial_offset_sec = xcTrialCol;
diagTable.xcorr_trial_corr = xcTrialCorrCol;
diagTable.spike_offset_sec = spikeCol;
diagTable.review_status = reviewStatus;
diagTable.review_reason = reviewReason;
diagTable.kept = keep;

outputTable = outputTable(keep, :);
end

function [status, reason] = reviewPaths(T, groupOfRow, manOff, manualOnly)
% 'Need Review' when the offset is unverified, a manual start is more than 0.5 s off the median of its
% recording (manual-only), VICON covers < 90 % of the movement window, or the finger marker of the active
% hand is missing in > 10 % of its frames.
    n = height(T);
    reason = strings(n, 1);
    src = string(T.align_source);
    sheet = manualOnly & ismember(src, ["manual", "manual_trial_median"]);
    for r = 1:n
        why = strings(0);
        if T.align_support(r) <= 1 && ~sheet(r) && T.total_frames(r) > 0
            why(end+1) = "unverified offset"; %#ok<AGROW>
        end
        if manualOnly && src(r) == "manual"
            dev = manOff(r) - median(manOff(groupOfRow == groupOfRow(r)), 'omitnan');
            if abs(dev) > 0.5
                why(end+1) = sprintf("manual start %+.2f s off its trial", dev); %#ok<AGROW>
            end
        end
        if T.vicon_coverage(r) == 0
            why(end+1) = "no VICON data"; %#ok<AGROW>
        elseif T.vicon_coverage(r) < 0.9
            why(end+1) = sprintf("VICON coverage %.0f %%", 100 * T.vicon_coverage(r)); %#ok<AGROW>
        end
        if T.total_frames(r) > 0
            m = "RFIN";
            if contains(lower(string(T.trial_hand{r})), "left"), m = "LFIN"; end
            c = find(strcmpi(T.labels{r}, m), 1);
            if ~isempty(c)
                miss = 100 * mean(any(isnan(T.position_data_interpolated{r}(:, c, :)), 1));
                if miss > 10
                    why(end+1) = sprintf("%s missing %.0f %%", m, miss); %#ok<AGROW>
                end
            end
        end
        reason(r) = strjoin(why, "; ");
    end
    status = repmat({'OK'}, n, 1);
    status(reason ~= "") = {'Need Review'};
end

function [off, mark] = pickManual(cand, cons)
% Per trial, the manual reading (trial start | movement start) nearest the consensus.
    off = cand(:, 1); mark = repmat("path_start", size(cand, 1), 1);
    if isnan(cons), return; end
    useMove = abs(cand(:, 2) - cons) < abs(cand(:, 1) - cons);
    off(useMove) = cand(useMove, 2); mark(useMove) = "move_start";
    mark(isnan(off)) = "";
end

function [cons, support] = consensusOffset(est, weight, tol, id)
% Offset agreed on by the most estimates (ties: higher summed weight). Estimates
% sharing an id (two readings of one manual entry) count once.
    est = est(:); weight = weight(:);
    if nargin < 4, id = (1:numel(est))'; end
    ok = ~isnan(est);
    est = est(ok); weight = weight(ok); id = id(ok);
    cons = NaN; support = 0;
    if isempty(est)
        return;
    end
    bestScore = -Inf;
    for i = 1:numel(est)
        near = abs(est - est(i)) <= tol;
        n = numel(unique(id(near)));
        score = n + 1e-3 * sum(weight(near));
        if score > bestScore
            bestScore = score;
            cons = median(est(near));
            support = n;
        end
    end
end

function dt = parseTime(s)
    s = string(s);
    dt = datetime(s, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSS');
    bad = isnat(dt) & strlength(s) > 0;
    if any(bad)
        dt(bad) = datetime(s(bad));
    end
end

function v = dayToVisit(day)
    if isnumeric(day)
        v = double(day(:));
        return;
    end
    v = str2double(regexprep(string(day(:)), '\D', ''));   % 'Day-2' -> 2
end
