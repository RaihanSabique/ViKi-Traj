% plot_review.m
% Review plots of every VICON recording whose offset is unverified (align_support <= 1 in
% vicon_alignment_diagnostics.csv), saved to review_plots/<pid>_group<g>_review.png. The first
% row shows the whole recording with the trials shaded at the offset used, then one row per trial:
%   left    finger speed (VICON) and game-object speed over game time at the offset used
%   middle  game-object trajectory RMov_path over the road drawn by the game (road_path, grey), screen px
%   right   VICON finger trajectory in the X-Y plane (mm) over the same window
%
%   matlab -batch "plot_review"

thisDir = fileparts(mfilename('fullpath'));
repoDir = fullfile(thisDir, '..', '..');
alignDir = fullfile(repoDir, 'outputs', 'game_vicon_aligned', 'reaching');
gameDir = fullfile(repoDir, 'data', 'game_data');
viconDir = fullfile(repoDir, 'data', 'vicon_data');
plotDir = fullfile(alignDir, 'review_plots');
if ~exist(plotDir, 'dir'), mkdir(plotDir); end

D = readtable(fullfile(alignDir, 'vicon_alignment_diagnostics.csv'), 'TextType', 'string');
review = unique(D(D.align_support <= 1 & ~ismissing(D.vicon_trial) & D.vicon_trial ~= "", {'pid', 'group_no'}), 'rows');

for i = 1:height(review)
    pid = char(review.pid(i));
    g = review.group_no(i);
    P = D(D.pid == review.pid(i), :);          % same row order as the sorted game table
    rowIdx = find(P.group_no == g);
    rows = P(rowIdx, :);

    G = load(fullfile(gameDir, [pid '.mat']), 'Reaching');
    gs = G.Reaching;
    if isfield(gs, 'Properties'), gs = rmfield(gs, 'Properties'); end
    game = struct2table(gs);
    [~, order] = sort(datetime(string(game.trial_start_time), 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSS'));
    game = game(order, :);
    V = load(fullfile(viconDir, [pid '_VICON.mat']));
    fn = fieldnames(V);
    vicon = V.(fn{1});
    vt = rows.vicon_trial(find(~ismissing(rows.vicon_trial) & rows.vicon_trial ~= "", 1));
    vname = extractBefore(vt, " | ");
    vr = vicon(find(string({vicon.trial}) == vname, 1));
    t = double(vr.time(:))';
    rate = double(vr.framerate);

    f = figure('Visible', 'off', 'Position', [0 0 1500 330 * (height(rows) + 1)]);
    tl = tiledlayout(height(rows) + 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('%s  %s  (group %d): offset %.2f s, UNVERIFIED', pid, vt, g, ...
        rows.consensus_offset_sec(1)), 'Interpreter', 'none');

    % Overview: whole VICON recording, both fingers, trials shaded at the offset used
    nexttile([1 3]); hold on;
    pathColor = [0.85 0.33 0.10; 0.00 0.45 0.74];        % right, left
    yl = 0;
    for m = {'RFIN', 'LFIN'}
        c = find(strcmpi(vr.labels, m{1}), 1);
        pv = squeeze(vr.position_data_interpolated(:, c, :));
        sv = movmean([NaN, sqrt(sum(diff(pv, 1, 2) .^ 2, 1)) * rate], 10, 'omitnan');
        yl = max(yl, prctile(sv, 99.5));
        plot(t, sv, '-', 'Color', pathColor(1 + strcmp(m{1}, 'LFIN'), :), 'DisplayName', [m{1} ' speed']);
    end
    ylim([0 yl]);
    for k = 1:height(rows)
        off = rows.vicon_offset_sec(k);
        if isnan(off), off = rows.consensus_offset_sec(k); end
        ts = double(game.RMov_time_step_sec{rowIdx(k)});
        x0 = rows.relative_start_sec(k) + min(ts) + off;      % movement window
        x1 = rows.relative_start_sec(k) + max(ts) + off;
        isLeft = contains(lower(rows.trial_hand(k)), 'left');
        patch([x0 x1 x1 x0], [0 0 yl yl], pathColor(1 + isLeft, :), 'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
        text(x0, 0.95 * yl, sprintf(' path %d (%s)', rows.trial_no(k), rows.trial_hand(k)), 'FontSize', 8);
    end
    xlim([min(t(1), rows.relative_start_sec(1) + rows.consensus_offset_sec(1)), ...
        max(t(end), x1)]);
    xline([t(1) t(end)], 'k--', {'VICON start', 'VICON end'}, 'HandleVisibility', 'off');
    xlabel('VICON time (s)'); ylabel('finger speed (mm/s)');
    legend('Location', 'northeastoutside');
    title(sprintf('Whole VICON trial (%.1f s); shaded = game paths at the offset used (game span %.1f s)', ...
        t(end) - t(1), x1 - (rows.relative_start_sec(1) + rows.consensus_offset_sec(1))), 'FontSize', 9);

    for k = 1:height(rows)
        r = rowIdx(k);
        grpStart = rows.relative_start_sec(k);
        xy = double(game.RMov_path{r});
        ts = double(game.RMov_time_step_sec{r}); ts = ts(:);
        n = min(size(xy, 1), numel(ts)); xy = xy(1:n, :); ts = ts(1:n);
        gtime = grpStart + ts;
        off = rows.vicon_offset_sec(k);
        if isnan(off), off = rows.consensus_offset_sec(k); end

        m = 'RFIN';
        if contains(lower(rows.trial_hand(k)), 'left'), m = 'LFIN'; end
        c = find(strcmpi(vr.labels, m), 1);
        pv = squeeze(vr.position_data_interpolated(:, c, :));
        win = t - off >= gtime(1) & t - off <= gtime(end);

        % speed over time
        nexttile; hold on;
        sv = movmean([NaN, sqrt(sum(diff(pv, 1, 2) .^ 2, 1)) * rate], 10, 'omitnan');
        dts = max(diff(ts), 1e-3);
        sg = movmean([0; sqrt(sum(diff(xy) .^ 2, 2)) ./ dts], 8);
        yyaxis left; plot(t(win) - off, sv(win), '-'); ylabel(sprintf('%s speed (mm/s)', m));
        yyaxis right; plot(gtime, sg, '-'); ylabel('game-object speed (px/s)');
        xlim([gtime(1) gtime(end)]); xlabel('game time (s)');
        title(sprintf('path %d (%s) | %s | manual off %.2f, xcorr %.2f (r=%.2f)', rows.trial_no(k), ...
            rows.trial_hand(k), fillmissing(rows.align_source(k), "constant", "dropped"), rows.manual_offset_sec(k), rows.xcorr_offset_sec(k), rows.xcorr_corr(k)), ...
            'Interpreter', 'none', 'FontSize', 8);
        if ~rows.kept(k)
            text(0.5, 0.5, 'DROPPED: outside VICON trial', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'Color', 'r');
        end

        % game-object trajectory on the road
        nexttile;
        rp = double(game.road_path{r});
        if ~isempty(rp), plot(rp(:, 1), rp(:, 2), '-', 'Color', [0.8 0.8 0.8]); hold on; end
        scatter(xy(:, 1), xy(:, 2), 4, ts, 'filled'); axis equal ij; colorbar;
        title('game RMov\_path (px, colour = time s)', 'FontSize', 8);

        % VICON finger trajectory
        nexttile;
        if any(win)
            scatter(pv(1, win), pv(2, win), 4, t(win) - off - grpStart, 'filled'); axis equal; colorbar;
        end
        xlabel('X (mm)'); ylabel('Y (mm)');
        title(sprintf('VICON %s X-Y plane, same window (colour = time s)', m), 'FontSize', 8);
    end
    out = fullfile(plotDir, sprintf('%s_group%d_review.png', pid, g));
    exportgraphics(f, out, 'Resolution', 80);
    close(f);
    fprintf('saved %s\n', out);
end
