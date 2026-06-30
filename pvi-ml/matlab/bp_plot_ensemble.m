clear
close all

ds_root = "D:\PviProject\datasets";

matName = "interps_long.mat";
matPath = fullfile(ds_root, matName);

% load data
load(matPath);
% bp_cross = [bp_main; bp_holdout];

bp_plot = bp;

%% plot ensemble

num_samples = size(bp_plot, 1);

T = 50;
close all
tvec = linspace(0, 1, T);

% preparing axe
fg = figure;

ax = axes; hold on
ax.TickLength = [0 0];
ax.TickDir = 'none';
ax.XLim = [0 1];
ax.XAxis.Visible = 0;
ax.YLim = [30 200];
ax.YTick = [30 115 200];
ax.PlotBoxAspectRatio = [2 1 1];

% plot
for k = 1:num_samples
    bpvec = bp_plot(k,:);

    plt = plot(ax, tvec, bpvec);
    plt.LineWidth = 0.1;
    plt.Color = 0.85*[1 1 1];
    plt.MarkerFaceColor = 'none';
    plt.MarkerEdgeColor = 'none';
end

%% plot mean

bpMEAN = mean(bp_plot,1);
bpDEV = std(bp_plot, [], 1);
kDEV = 1;

plot(tvec, bpMEAN, '-r', 'LineWidth', 1);
plot(tvec, bpMEAN + kDEV*bpDEV, '-k', 'LineWidth', 1);
plot(tvec, bpMEAN - kDEV*bpDEV, '-k', 'LineWidth', 1);


%%
fgName = "bp_interps_long";
fgPath = fullfile(ds_root, fgName);
% exportgraphics(fg, fgPath + ".pdf", ...
%     "BackgroundColor","none", ...
%     "ContentType", "vector", ...
%     "Resolution", 300);