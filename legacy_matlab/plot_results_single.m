function plot_results_single(mlrp, fg)
    arguments
        mlrp (1,1) MLTRAININGREPORT
        fg (1,1) matlab.ui.Figure = gcf
    end

    data = table2array(mlrp.results);
    tl = mlrp.session;

    num_cols = size(data,2);
    predictions = data(:,1:num_cols/2);
    targets = data(:,(num_cols/2 + 1):end);

    clf(fg);
    
    layout = tiledlayout(fg, 3, 4);
    locs = [1 2 3 4 5 6 7 8 9 11];
    rows = ones(1, 10);
    cols = [1 1 1 1 1 1 1 1 2 2];
    rc = [rows(:), cols(:)];
    for k = 1:numel(locs)
        ax(k) = nexttile(layout, locs(k), rc(k,:));
    end

    layout.Padding = "tight";
    layout.TileSpacing = "tight";

    layout.Title.String = tl;
    layout.Title.FontWeight = 'bold';

    plot_correlation(sbp(predictions), sbp(targets), "SBP", ax(1));
    plot_bland_altman(sbp(predictions), sbp(targets), "SBP", ax(2));
    plot_histogram_error(sbp(predictions), sbp(targets), ax(3));
    plot_histogram_compare(sbp(predictions), sbp(targets), "SBP", ax(4));

    plot_correlation(dbp(predictions), dbp(targets), "DBP", ax(5));
    plot_bland_altman(dbp(predictions), dbp(targets), "DBP", ax(6));
    plot_histogram_error(dbp(predictions), dbp(targets), ax(7));
    plot_histogram_compare(dbp(predictions), dbp(targets), "DBP", ax(8));

    ax(9) = nexttile(layout, locs(9), rc(9,:));
    plot_ensemble(data, "predictions", ax(9));
    
    ax(10) = nexttile(layout, locs(10), rc(10,:));
    plot_ensemble(data, "targets", ax(10));

    for k = 1:numel(ax)
        ax(k).Box = "off";
        ax(k).TickDir = "None";
        ax(k).TickLength = [0 0];
    end

    for k = 1:8
        ax(k).PlotBoxAspectRatio = [1, 1, 1];
    end
end

%% Supporting functions
function plot_correlation(predictions, targets, bptag, ax)
    arguments
        predictions {mustBeNumeric}
        targets {mustBeNumeric}
        bptag (1,1) string = "BP";
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end

    if nargin < 4 || isempty(ax)
        ax = gca;
    end

    ax.Tag = "correlation";
    ax.XLabel.String = sprintf("True %s (mm Hg)", bptag);
    ax.YLabel.String = sprintf("Predicted %s (mm Hg)", bptag);

    cla(ax);
    hold(ax,"on");

    % Convert to column vectors
    predictions = predictions(:);
    targets = targets(:);

    % Perform linear regression (polyfit equivalent)
    coeffs = polyfit(targets, predictions, 1);
    
    % Calculate limits for plot
    data = [predictions(:); targets(:)];
    [dL, dH] = get_bp_range(data, []);
    dL = dL - mod(dL, 2);  % Make even if odd
    dH = dH + mod(dH, 2);  
    dM = mean([dL, dH]);
    
    % Scatter plot
    sct = scatter(ax, targets, predictions);
    sct.Marker = "o";
    sct.SizeData = 12;
    sct.MarkerFaceColor = "#29AAE1";
    sct.MarkerFaceAlpha = 0.15;
    sct.MarkerEdgeColor = "None";
    hold on;
    
    lw = 1.0;

    % Plot reference line
    plt45 = plot(ax, [dL dH], [dL dH], '--');
    plt45.Color = 'black';
    plt45.LineWidth = 0.5;

    % Plot regression line
    plt = plot(ax, targets, polyval(coeffs, targets), '-');
    plt.Color = 'black';
    plt.LineWidth = lw;

    % Calculate Pearson correlation coefficient and p-value
    [r, pv] = corr(predictions, targets, 'Type', 'Pearson');
    cc_func = @(X,Y) 2*r*std(X)*std(Y)/(std(X)^2 + std(Y)^2 + (mean(X) - mean(Y))^2);
    cc = cc_func(predictions, targets);
    
    if pv < 0.001
        pv_string = "p < 0.001";
    elseif pv < 0.01
        pv_string = "p < 0.01";
    elseif pv < 0.05
        pv_string = "p < 0.05";
    else
        pv_string = num2str(round(pv,4));
    end

    r2_string = sprintf("r^2 = %.2f", round(r^2, 2));
    cc_string = sprintf("cc = %.2f", round(cc, 2));
    
    str = join([r2_string, cc_string, pv_string],"\n");
    txt = text(ax, 0.05, 0.95, sprintf(str), ...
         'HorizontalAlignment', 'left',...
         'VerticalAlignment', 'top',...
         'Units','normalized', ...
         'Interpreter','tex');
    
    % Set axis limits and ticks
    ax.XLim = [dL, dH];
    ax.XTick = [dL, dM, dH];
    ax.YLim = ax.XLim;
    ax.YTick = ax.XTick;
end

function plot_bland_altman(predictions, targets, bptag, ax)
    arguments
        predictions {mustBeNumeric}
        targets {mustBeNumeric}
        bptag (1,1) string = "BP";
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end

    if nargin < 3 || isempty(ax)
        ax = gca;
    end

    ax.Tag = "BA";
    ax.XLabel.String = 'Means (mm Hg)';
    ax.YLabel.String = 'Difference (mm Hg)';

    cla(ax);
    hold(ax,"on");
    
    % Convert inputs to column vectors
    predictions = predictions(:);
    targets = targets(:);
    
    % Mean and difference between prediction and test
    mean12 = mean([predictions, targets], 2);
    err = predictions - targets;   % Difference
    md = mean(err);            % Mean of the difference
    sd = std(err);             % Standard deviation of the difference

    CI_low = prctile(err, 2.5);
    CI_high = prctile(err, 97.5);
    
    % Calculate x-axis limits
    [xLow, xHigh] = get_bp_range(mean12, []);    
    xHigh = xHigh + mod(xHigh, 2);  % Make even if odd
    xLow = xLow - mod(xLow, 2);  % Make even if odd
    xMid = mean([xLow, xHigh]);
    
    % Calculate y-axis limits
    yLow = floor(md - 5 * sd);
    yHigh = ceil(md + 5 * sd);
    
    % Scatter plot of means vs differences
    sct = scatter(ax, mean12, err);
    sct.Marker = "o";
    sct.SizeData = 12;
    sct.MarkerFaceColor = "#29AAE1";
    sct.MarkerFaceAlpha = 0.15;
    sct.MarkerEdgeColor = "None";
    hold on;
    
    lw = 0.5;
    % Plot horizontal lines for mean difference and confidence intervals
    % Mean difference and upper, lower CI lines
    yline(ax, md, '-k', 'LineWidth', lw);
    yline(ax, CI_high, '--k', 'LineWidth', lw);
    yline(ax, CI_low, '--k', 'LineWidth', lw);
    
    % Annotate the plot
    text(ax, xHigh, md - 0.05 * sd, num2str(round(md, 2)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');
    text(ax, xHigh, CI_low - 0.05 * sd, num2str(round(CI_low, 2)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');
    text(ax, xHigh, CI_high - 0.05 * sd, num2str(round(CI_high, 2)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');    
    
    % Set axis limits and ticks
    ax.XLim = [xLow, xHigh];
    ax.XTick = [xLow, xMid, xHigh];
    ax.YLim = [yLow, yHigh];
    ax.YTick = [yLow, 0, yHigh];
end

function plot_histogram_error(predictions, targets, ax)
    arguments
        predictions {mustBeNumeric}
        targets {mustBeNumeric}
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end

    if nargin < 3 || isempty(ax)
        ax = gca;
    end

    ax.Tag = "histogram";
    ax.XLabel.String = 'Absolute error (mm Hg)';
    ax.YLabel.String = 'Occurrences';
    
    cla(ax);
    hold(ax,"on");
    
    % Convert inputs to column vectors
    predictions = predictions(:);
    targets = targets(:);

    % Calculate prediction error
    err = predictions - targets;
    err = abs(err);

    % Create histogram
    low = 0;
    high = 100;
    delta = 1;
    edges = low:delta:high;

    h = histogram(ax, err, edges);

    h.FaceColor = "#29AAE1";
    h.FaceAlpha = 0.75;
    h.EdgeColor = "None";
    
    % Calculate x and y axis limits
    xL = floor(min(h.BinEdges));
    xH = ceil(max(h.BinEdges));
    xM = mean([xL, xH]);
    
    yL = 0;
    yH = max(h.BinCounts);
    yH = yH + mod(yH, 2);  % Make even if odd
    yM = mean([yL, yH]);

    % Set axis limits and ticks
    ax.XLim = [xL, xH];
    ax.XTick = [xL, xM, xH];
    ax.YLim = [yL, yH];
    ax.YTick = [yL, yM, yH];
    
    % compute metrics for annotation
    bounds = [5, 10, 15];
    func = @(e, b) sum(abs(e) < b)/numel(e);
    
    ratios = arrayfun(@(b) func(err,b) ,bounds);
    percents = round(ratios*100);
    mae = round(mean(err), 2);
    sdae = round(std(err), 2);
    
    str1 = sprintf('MEAN: %.2f\nSDAE: %.2f', mae, sdae);
    str2 = arrayfun(@(b, p) sprintf('%d-tol: %d%%', b, p), bounds, percents,...
        'UniformOutput', false);
    
    str = [{str1}, str2];

    lw = 1.0;
    cs = cumsum(h.Values);
    cs = [0 cs]/max(cs)*yH;
    plt = plot(ax, h.BinEdges, cs, '-k', ...
        'LineWidth', lw);

    plt = plot(ax, bounds, ratios*yH, 'or');
    plt.MarkerFaceColor = plt.Color;
    plt.MarkerEdgeColor = "none";

    txt = text(ax, 0.95, 0.95, str, ...
        'HorizontalAlignment', 'right',...
        'VerticalAlignment', 'top',...
        'Units','normalized');
end

function plot_histogram_compare(predictions, targets, bptag, ax)
    arguments
        predictions {mustBeNumeric}
        targets {mustBeNumeric}
        bptag (1,1) string = "BP";
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end

    if nargin < 4 || isempty(ax)
        ax = gca;
    end

    ax.Tag = "histogram";
    ax.XLabel.String = sprintf("%s (mm Hg)", bptag);
    ax.YLabel.String = 'Occurrences';

    cla(ax);
    hold(ax,"on");
    
    % Convert inputs to column vectors
    predictions = predictions(:);
    targets = targets(:);

    % Create histogram
    % num_bins = 50;
    low = floor(min([predictions; targets]));
    high = ceil(max([predictions; targets]));
    delta = 1;
    edges = low:delta:high;

    h1 = histogram(ax, predictions, edges);
    h1.DisplayName = "Predicted";
    h1.FaceAlpha = 0.5;
    h1.FaceColor = "#29AAE1";
    h1.EdgeColor = "None";

    h2 = histogram(ax, targets, edges);
    h2.DisplayName = "True";
    h2.FaceAlpha = 0.5;
    h2.FaceColor = "#EB2027";
    h2.EdgeColor = "None";

    E = sort([h1.BinEdges(:); h2.BinEdges(:)]);

    % Calculate x and y axis limits
    [xL, xH] = get_bp_range(E, []);
    xL = xL - mod(xL, 2);  % Make even if odd
    xH = xH + mod(xH, 2);  
    xM = mean([xL, xH]);
        
    yL = 0;
    yH = max([h1.BinCounts(:); h2.BinCounts(:)]);
    yH = yH + mod(yH, 2);  % Make even if odd
    yM = mean([yL, yH]);

    % Set axis limits and ticks
    ax.XLim = [xL, xH];
    ax.XTick = [xL, xM, xH];
    ax.YLim = [yL, yH];
    ax.YTick = [yL, yM, yH];

    lgd = legend(ax);
    lgd.Location = 'northeast';
    lgd.Orientation = 'vertical';
    lgd.Box = 'off';

    % compute 1D wasserstein distance for annotation
    wd = compute_wd_1d(predictions, targets);

    str = sprintf('WD: %.2f', round(wd, 2));
    
    txt = text(ax, 0.95, 0.12, str, ...
        'HorizontalAlignment', 'right',...
        'VerticalAlignment', 'baseline',...
        'Units','normalized');

end

function plot_ensemble(data, tag, ax)
    arguments
        data {mustBeNumeric}
        tag (1,1) string
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end
    
    if nargin < 3 || isempty(ax)
        ax = gca;
    end

    ax.Tag = "ensemble";

    num_cols = size(data,2);
    predictions = data(:,1:num_cols/2);
    targets = data(:,(num_cols/2 + 1):end);
    
    if strcmpi(tag, "predictions")
        data = predictions;
        label = "Predicted BP (mm Hg)";
    elseif strcmpi(tag, "targets")
        data = targets;
        label = "True BP (mm Hg)";
    else
        error("something ain't right")
    end
    
    ax.YLabel.String = label;

    cla(ax);
    hold(ax,"on");

    tvec = 1:size(data,2);
    sMEAN = mean(data, 1);
    sDEV = std(data, [], 1);
    kDEV = 1;

    plts = plot(tvec, data, '-');

    color = [.75*[1 1 1] .1]; % last value is alpha
    arrayfun(@(plt) set(plt, "Color", color), plts);
    arrayfun(@(plt) set(plt, "LineWidth", 0.1), plts);

    lw = 0.5;
    plot(tvec, sMEAN, '-k', 'LineWidth', lw);
    plot(tvec, sMEAN + kDEV*sDEV, '--k', 'LineWidth', lw);
    plot(tvec, sMEAN - kDEV*sDEV, '--k', 'LineWidth', lw);

    % dPad = 0.075; % padding
    % dLow = floor(min(data(:)));
    % dHigh = ceil(max(data(:)));
    % dLow = floor(dLow - dPad * (dHigh - dLow));
    % dHigh = ceil(dHigh + dPad * (dHigh - dLow));

    yL = 30;
    yH = 200;
    yM = round(mean([yL, yH]), 2);

    if strcmpi(tag, "predictions")
        err = predictions - targets;

        mae = sum(abs(err),2)/size(err,2);
        rmse = sqrt(sum(err.^2,2)/size(err,2));

        amae = round(sum(mae)/size(err,1), 2);
        armse = round(sum(rmse)/size(err,1), 2);
        str = sprintf('AMAE: %.2f\nARMSE: %.2f', amae, armse);
        
        txt = text(ax, 0.95, 0.95, str, ...
            'HorizontalAlignment', 'right',...
            'VerticalAlignment', 'top',...
            'Units','normalized');

    end

    ax.YLim = [yL, yH];
    ax.YTick = [yL, yM, yH];
    ax.XLim = [tvec(1) tvec(end)];
    ax.XAxis.Visible = false;

end

%% support functions

function bp_out = sbp(bp_in)
    bp_out = max(bp_in, [], 2);
end

function bp_out = dbp(bp_in)
    bp_out = min(bp_in, [], 2);
end

function [dLow, dHigh] = get_bp_range(data, bptag, padding)
    arguments
        data {mustBeNumeric}
        bptag = []
        padding (1,1) {mustBeNumeric} = 0
    end
    
    if strcmpi(bptag,"dbp")
        dLow = 30;
        dHigh = 130;

    elseif strcmpi(bptag,"sbp")
        dLow = 60;
        dHigh = 200;

    elseif strcmpi(bptag, "bp")
        dLow = 30;
        dHigh = 200;

    else % auto mode
        data = data(:);
        dLow = floor(min(data));
        dHigh = ceil(max(data));
    end

    if padding>0
        dLow = floor(dLow - padding * (dHigh - dLow));
        dHigh = ceil(dHigh + padding * (dHigh - dLow));
    end

end

function wd = compute_wd_1d(predictions, targets)
    % Wasserstein-1 distance between two univariate distributions
    % See https://gchron.copernicus.org/articles/5/263/2023/#bib1.bibx11
    % for interpretation
    low = floor(min([predictions; targets])) - 0.5;
    high = ceil(max([predictions; targets])) + 0.5;
    dx = 0.1;

    edges = low:dx:high;
    p1 = histcounts(predictions, edges)/numel(predictions);
    p2 = histcounts(targets, edges)/numel(targets);

    c1 = cumsum(p1);
    c2 = cumsum(p2);

    wd = sum(abs(c1 - c2)*dx);
end