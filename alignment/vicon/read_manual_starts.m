function M = read_manual_starts(xlsxPath, participantId)
% read_manual_starts Manual VICON start times of the trials of the reaching game.
%
%   M = read_manual_starts(xlsxPath, participantId)
%
%   Reads the 'Frame Times' sheet of the marker-labelling workbook (one row per participant).
%   The start of trial p of VICON recording t in visit v is in the column named like
%   'V1_Reaching_Path2 Trial 3' (seconds), followed by its VICON frame (100 Hz). Column names
%   are compared in lower case without spaces or punctuation.
%
%   M  2 x 3 x 4 array (visit x VICON recording x trial) of manual start times in seconds from
%      the start of the VICON recording: the seconds, else frame/100. NaN when the entry is
%      empty, 'x' or not positive, or the participant has no row in the sheet.

M = nan(2, 3, 4);

persistent cachedPath cachedCells
if isempty(cachedPath) || ~strcmp(cachedPath, xlsxPath)
    C = readcell(xlsxPath, 'Sheet', 'Frame Times');
    C(cellfun(@(v) all(ismissing(v)), C)) = {''};
    cachedCells = C;
    cachedPath = xlsxPath;
end
C = cachedCells;

rowIdx = find(strcmpi(strtrim(string(C(:,1))), participantId), 1);
if isempty(rowIdx)
    warning('reaching:manual', '%s: no row in Frame Times sheet.', participantId);
    return;
end

canon = @(s) regexprep(lower(char(string(s))), '[^a-z0-9]', '');
hdr = cellfun(canon, C(1,:), 'UniformOutput', false);

for v = 1:2
    for tr = 1:3
        for p = 1:4
            col = find(strcmp(hdr, canon(sprintf('V%d_Reaching_Path%d Trial %d', v, p, tr))), 1);
            if isempty(col)
                continue;
            end
            sec = toNumber(C{rowIdx, col});
            if isnan(sec) && col < size(C,2)
                frame = toNumber(C{rowIdx, col+1});
                sec = frame / 100;
            end
            M(v, tr, p) = sec;
        end
    end
end
end

function x = toNumber(v)
    if isnumeric(v) && isscalar(v)
        x = double(v);
    else
        x = str2double(strtrim(char(string(v))));   % 'x', '' -> NaN
    end
    if ~isfinite(x) || x <= 0
        x = NaN;
    end
end
