function ax = fix_axis(ax, XData, YData)
    arguments
        ax (1,1) matlab.graphics.axis.Axes;
        XData {mustBeNumeric} = []
        YData {mustBeNumeric} = []
    end
    
    if isempty(XData)
        XData = get_plot_data(ax, 'XData');
    end
    if isempty(YData)
        YData = get_plot_data(ax, 'YData');
    end
    
    [xLim, xTick] = format_ticks(XData, 1);
    [yLim, yTick] = format_ticks(YData, 1);
    
    ax.XLim = xLim;
    ax.XTick = xTick;

    ax.YLim = yLim;
    ax.YTick = yTick;

    ax.TickDir = 'out';
end

%% support function
function [lims, ticks] = format_ticks(data, decimals)
    arguments
        data double
        decimals (1,1) = 1;
    end

    factor = 10^decimals;

    [low, high] = bounds(data);
    low = floor(low*factor)/factor;
    high = ceil(high*factor)/factor;

    % is_even = @(num) mod(num, 2) == 0;
    is_whole = @(num) rem(num, 1) == 0;

    if all(is_whole(data), "all") % if whole number
        % Make even if odd
        high = high + mod(high, 2);
        low = low - mod(high, 2);
    else
        ;
    end

    lims = [low high];
    ticks = sort([lims, mean(lims)]);
end

function data = get_plot_data(ax, fn)
    arguments
        ax (1,1) matlab.graphics.axis.Axes;
        fn string
    end
    data = [];
    for k = 1:numel(ax.Children)
        if ~isprop(ax.Children(k), fn)
            continue;
        else
            tmp = ax.Children(k).(fn);
            data = [data(:); tmp(:)];
        end
    end
end