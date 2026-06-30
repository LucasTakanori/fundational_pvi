clear
close all

ds_root = "D:\PviProject\datasets";

matName = "interps_main";
matPath = fullfile(ds_root, matName) + ".mat";

load(matPath)

%% extract channels

zHP = mean(zHP, 3)*1e3;
zLP = mean(zLP, 3)*1e3;

%% plot ensemble

close all
fg = figure;
ax1 = axes;
plot_interps(bp, ax1);

ax1.YLim = [30 200];
ax1.YTick = sort([ax1.YLim, mean(ax1.YLim)]);

fgName = join(["fg", matName, "bp"], "_");
fgPath = fullfile(ds_root, fgName);
exportgraphics(fg, fgPath + ".pdf", "ContentType", "vector", "Resolution", 600);

fg = figure;
ax2 = axes;
plot_interps(zHP, ax2);

ax2.YLim = 15*[-1 1];
ax2.YTick = sort([ax2.YLim, mean(ax2.YLim)]);

fgName = join(["fg", matName, "zHP"], "_");
fgPath = fullfile(ds_root, fgName);
exportgraphics(fg, fgPath + ".pdf", "ContentType", "vector", "Resolution", 600);