function scatter_regress(ax, x, y, color)
    arguments
        ax (1,1) matlab.graphics.axis.Axes;
        x {mustBeNumeric} = []
        y {mustBeNumeric} = []
        color (1,3) = [0 0 0]
    end
    
    [x, idx] = sort(x);
    y = y(idx);

    p = polyfit(x,y,1);
    yfit = polyval(p, x);
    c95 = polyconf(p, x);

    plt = plot(ax, x, y, 'o');
    plt.Color = color;
    plt.MarkerFaceColor = plt.Color;
    plt.MarkerEdgeColor = "none";
    plt.MarkerSize = 3.5;

    fplt = plot(ax, x, yfit);
    fplt.Color = color;
    fplt.LineWidth = 0.5;

    xconf = [x; flip(x)];
    yconf = [(yfit+c95); flip(yfit-c95)];

    [r, pv] = corr(x, y, 'Type', 'Pearson');
    r_string = "r = " + string(round(r,2));

    % Annotate correlation and p-value
    if pv < 0.05
        pv_string = "p < 0.05";
    else
        pv_string = "p = " + string(round(pv,2));
    end

    str = sprintf(join([r_string, pv_string],"\n"));

    txt1 = text(ax, 1, 1, str, ...
        'Unit','normalized',...
        'HorizontalAlignment','right',...
        'VerticalAlignment','top');

    fe = fill(ax, xconf, yconf, [1 1 1]);
    fe.EdgeColor = "none";
    fe.FaceColor = color;
    fe.FaceAlpha = 0.1;
end