function plot_interps(data, ax)
    arguments
        data double
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end

    hold(ax, "on");

    tvec = linspace(0, 1, 50);
    plts = plot(ax, tvec, data);
    
    for k = 1:size(data,1)
        plts(k).LineWidth = 0.1;
        plts(k).Color = [.75*[1 1 1] 0.1];
    end

    sMEAN = mean(data, 1);
    sDEV = std(data, [], 1);

    plot(ax, tvec, sMEAN, '-k', 'LineWidth', 0.5);
    plot(ax, tvec, sMEAN + sDEV, '--k', 'LineWidth', 0.5);
    plot(ax, tvec, sMEAN - sDEV, '--k', 'LineWidth', 0.5);

    hold(ax, "off");

    ax.Box = 0;
    ax.TickDir = "none";
    ax.XAxis.Visible=0;
end