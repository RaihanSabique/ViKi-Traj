function [offsetSec, info] = xcorr_offset(viconTime, positionData, labels, groupTbl, varargin)
% xcorr_offset VICON-to-game clock offset by matching finger-marker and game-object speed.
%
%   [offsetSec, info] = xcorr_offset(viconTime, positionData, labels, groupTbl, ...)
%
%   The game object follows the finger on the screen, so the speed of the finger marker of the
%   active hand (RFIN / LFIN) and the speed of the game object (RMov_path) have the same time
%   course. The offset that maximises the Pearson correlation of the two speeds is searched over
%   every frame shift (FFT cross-correlation):
%       vicon_time = game_time_relative_to_trial1_start + offset
%
%   groupTbl  game rows of one VICON recording with variables relative_start_sec, trial_hand,
%             RMov_path (cell, n x 2 px) and RMov_time_step_sec (cell, s).
%
%   Name-value options:
%     'Smooth'      moving-average window of both speeds (s), default 0.1
%     'MinOverlap'  fraction of the game samples that must fall inside the VICON recording,
%                   default 0.5
%
%   offsetSec                 best offset of all trials together (one clock per recording)
%   info.corr, info.peakRatio correlation at offsetSec, and the best correlation more than 1 s
%                             away divided by it (near 1 = ambiguous match)
%   info.pathOffset, info.pathCorr, info.pathPeakRatio
%                             the same for each trial matched on its own (NaN if not computable)

p = inputParser;
p.addParameter('Smooth', 0.1, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('MinOverlap', 0.5, @(x) isnumeric(x) && isscalar(x) && x > 0 && x <= 1);
p.parse(varargin{:});

t = double(viconTime(:))';
rate = 1 / median(diff(t));
win = max(1, round(p.Results.Smooth * rate));
nPaths = height(groupTbl);

offsetSec = NaN;
info = struct('corr', NaN, 'peakRatio', NaN, ...
    'pathOffset', nan(nPaths, 1), 'pathCorr', nan(nPaths, 1), 'pathPeakRatio', nan(nPaths, 1));

% VICON finger speed of each hand (NaN -> 0 with a validity mask)
[vR, wR] = fingerSpeed(positionData, labels, 'RFIN', rate, win);
[vL, wL] = fingerSpeed(positionData, labels, 'LFIN', rate, win);

% Game-object speed of each trial on the VICON frame grid (game time)
gIdx = cell(nPaths, 1); gS = cell(nPaths, 1); left = false(nPaths, 1);
for k = 1:nPaths
    xy = groupTbl.RMov_path{k};
    ts = groupTbl.RMov_time_step_sec{k};
    n = min(size(xy, 1), numel(ts));
    if n < 10, continue; end
    xy = double(xy(1:n, :)); ts = double(ts(1:n)); ts = ts(:);
    [ts, iu] = unique(ts); xy = xy(iu, :);
    tg = (ceil(ts(1) * rate):floor(ts(end) * rate)) / rate;
    if numel(tg) < 10, continue; end
    pg = interp1(ts, xy, tg, 'linear');
    sg = movmean([0, sqrt(sum(diff(pg).^2, 2))' * rate], win);
    gIdx{k} = round((groupTbl.relative_start_sec(k) + tg) * rate);   % 0-based game frame
    gS{k} = sg;
    left(k) = contains(lower(char(string(groupTbl.trial_hand{k}))), 'left');
end
use = find(~cellfun(@isempty, gIdx))';
if isempty(use)
    return;
end
Lg = max(cellfun(@(x) x(end), gIdx(use))) + 1;
N = numel(t);
L = 2 ^ nextpow2(N + Lg);

% Per-trial sums for the Pearson r at every shift s (frames):
% game frame j lands on VICON frame j + s, offset = t(1) + s / rate
F = @(x) fft(x, L);
FvR = F(vR); FvR2 = F(vR .^ 2); FwR = F(wR);
FvL = F(vL); FvL2 = F(vL .^ 2); FwL = F(wL);
xc = @(Fa, b) real(ifft(Fa .* conj(F(b))));
stats = cell(nPaths, 1);
for k = use
    G = zeros(1, Lg); M = zeros(1, Lg);
    G(gIdx{k} + 1) = gS{k}; M(gIdx{k} + 1) = 1;
    if left(k)
        Fv = FvL; Fv2 = FvL2; Fw = FwL;
    else
        Fv = FvR; Fv2 = FvR2; Fw = FwR;
    end
    stats{k} = struct('n', xc(Fw, M), 'sv', xc(Fv, M), 'svv', xc(Fv2, M), ...
        'sg', xc(Fw, G), 'sgg', xc(Fw, G .^ 2), 'svg', xc(Fv, G), 'total', numel(gIdx{k}));
end
shifts = [0:(N - 1), (-(L - N)):-1];    % circular index -> shift
offsets = t(1) + shifts / rate;

for k = use
    [info.pathOffset(k), info.pathCorr(k), info.pathPeakRatio(k)] = pickBest(stats(k), offsets, p.Results.MinOverlap);
end
[offsetSec, info.corr, info.peakRatio] = pickBest(stats(use), offsets, p.Results.MinOverlap);
end

function [best, rBest, ratio] = pickBest(S, offsets, minOverlap)
    tot = @(f) sum(cell2mat(cellfun(@(s) s.(f), S(:), 'UniformOutput', false)), 1);
    n = tot('n'); sv = tot('sv'); svv = tot('svv'); sg = tot('sg'); sgg = tot('sgg'); svg = tot('svg');
    total = sum(cellfun(@(s) s.total, S));
    n = round(n);
    den = sqrt(max(n .* svv - sv .^ 2, 0) .* max(n .* sgg - sg .^ 2, 0));
    r = (n .* svg - sv .* sg) ./ den;
    r(n < minOverlap * total | n < 100 | den <= 0) = NaN;    % at least 100 frames (1 s) of overlap
    [rBest, j] = max(r);
    if isnan(rBest)
        best = NaN; ratio = NaN;
        return;
    end
    best = offsets(j);
    ratio = max(r(abs(offsets - best) > 1), [], 'omitnan') / rBest;
end

function [s, w] = fingerSpeed(positionData, labels, marker, rate, win)
    c = find(strcmpi(labels, marker), 1);
    N = size(positionData, 3);
    if isempty(c)
        s = zeros(1, N); w = zeros(1, N);
        return;
    end
    pv = squeeze(positionData(:, c, :));                 % 3 x N
    s = [NaN, sqrt(sum(diff(pv, 1, 2) .^ 2, 1)) * rate];
    s = movmean(s, win, 'omitnan');
    w = double(~isnan(s));
    s(isnan(s)) = 0;
end
