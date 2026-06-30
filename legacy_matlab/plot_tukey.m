function plot_tukey(ydata, center, coverage, ax)
    arguments
        ydata (1,:) double
        center (1,1) double = 0
        coverage = []
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end

    space = 1;
    bwidth = 0.75*space;
    wlength = 0.5*bwidth;
    lw = 0.5;

    if isempty(coverage)
        coverage = 100;
    end
    mask1 = ydata >= prctile(ydata, 50 - coverage/2);
    mask2 = ydata <= prctile(ydata, 50 + coverage/2);
    ydata = ydata(mask1 & mask2);

    md = median(ydata);
    p25 = prctile(ydata, 25); % first quartile
    p75 = prctile(ydata, 75); % third quartile
    wLow = p25 - 1.5*iqr(ydata);
    wHigh = p75 + 1.5*iqr(ydata);

    wLow = min(ydata(ydata >= wLow));
    wHigh = max(ydata(ydata <= wHigh));

    % wLow = max(min(ydata), wLow);
    % wHigh = min(max(ydata), wHigh);

    % wLow = prctile(ydata, 2.5);
    % wHigh = prctile(ydata, 97.5);

    % plot scatter
    % x = center*ones(size(data));
    % sct = scatter(x, data, 'ok');
    % sct.SizeData = 10;
    % sct.MarkerEdgeColor = 'none';
    % sct.MarkerFaceColor = 'black';
    % sct.MarkerFaceAlpha = 0.15;

    % plot median
    mx = center + 0.5*bwidth*[-1 1];
    plot(ax, mx, [md md], '-r', 'LineWidth', 1.5);

    hold(ax,"on");

    % plot box
    px = center + 0.5*bwidth*[-1 1 1 -1];
    py = [p25 p25 p75 p75];
    p = patch(ax, px,py, 'white');
    p.FaceColor = 'none';
    p.EdgeColor = 'black';
    p.LineWidth = lw;

    % plot whiskers
    wx = center + 0.5*wlength*[-1, 1];
    plot(ax, wx, wLow*[1 1], '-k', 'LineWidth', lw);
    plot(ax, wx, wHigh*[1 1], '-k', 'LineWidth', lw);
    plot(ax, center*[1 1], [p75 wHigh], '-k', 'LineWidth', lw);
    plot(ax, center*[1 1], [p25 wLow], '-k', 'LineWidth', lw);
end
