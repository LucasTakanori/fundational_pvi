function plts = plot_histograms(data, edges, ax)
    arguments
        data double
        edges double = [];
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end

    if isempty(edges)
        edges = 100; % default 100 bins
    end

    dims = size(data);

    if dims(1) < dims(2)
        data = data';
    end
    
    colors = colororder("glow12");

    plts = [];
    xL = []; xH = [];
    yL = 0; yH = [];

    for k = 1:size(data, 2)
        if k > 1
            hold on
        end

        series = data(:, k);
        h = histogram(ax, series, edges);
        
        h.FaceColor = colors(k,:);
        h.FaceAlpha = 0.75;
        h.EdgeColor = "None";
        
        xL = min([xL min(h.BinEdges)]);
        xH = max([xH max(h.BinEdges)]);
        yH = max([yH max(h.BinCounts)]);
        
        sMEAN = round(mean(series)*100)/100;
        sSTD = round(std(series)*100)/100;

        txt = sprintf("MEAN:%.2f \t\t SD:%.2f", sMEAN, sSTD);
        xl = xline(ax, sMEAN, '--k', txt);
        % xl.LabelHorizontalAlignment = 'center';
        xl.FontSize = 8;

        plts = [plts h];

    end
    
    hold off

    % Make even if odd
    xL = floor(xL);
    xL = xL - mod(xL, 2);

    xH = ceil(xH);
    xH = xH + mod(xH, 2);

    order = floor(log10(yH));
    yH = ceil(yH/10^order)*(10^order);

    % Set axis limits and ticks
    ax.XLim = [xL, xH];
    ax.YLim = [yL, yH];
    
    ax.XTick = sort([ax.XLim mean(ax.XLim)]);
    ax.YTick = sort([ax.YLim mean(ax.YLim)]);

    if order >= 12
        ax.YTickLabel = string(ax.YTick/1e12) + "T";
    elseif order >= 9
        ax.YTickLabel = string(ax.YTick/1e9) + "B";
    elseif order >= 6
        ax.YTickLabel = string(ax.YTick/1e6) + "M";
    elseif order >= 3
        ax.YTickLabel = string(ax.YTick/1e3) + "K";
    else
        ax.YTickLabel = string(ax.YTick);
    end

    ax.TickDir = "out";
    ax.Box = 0;
end