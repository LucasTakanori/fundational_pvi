function plot_violin(ydata, center, coverage, ax)
    arguments
        ydata (1,:) double
        center (1,1) double = 0
        coverage = []
        ax (1,1) matlab.graphics.axis.Axes = gca;
    end

    space = 1;
    bwidth = 0.45*space;
    lw = 0.5;

    if isempty(coverage)
        coverage = 100;
    end
    mask1 = ydata >= prctile(ydata, 50 - coverage/2);
    mask2 = ydata <= prctile(ydata, 50 + coverage/2);
    ydata = ydata(mask1 & mask2);

    [x, y] = kde(ydata,"NumPoints", 250);
    x = x/max(x);

    X = [x(:); flip(-x(:))];
    Y = [y(:); flip(y(:))];

    X = X*bwidth + center;

    hold(ax, "on");

    % plot distribution
    fe = patch(ax, X, Y, 'white');
    fe.LineWidth = lw;
    fe.EdgeColor = "black";
    fe.FaceAlpha = 0;
end
