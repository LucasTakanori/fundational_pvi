clear
close all

ds_root = "D:\PviProject\datasets";

ds_subdir = fullfile(ds_root,"holdout");

interps_dir = fullfile(ds_subdir,"_stats");
interps_subject = fullfile(interps_dir,"subjects");

load(fullfile(ds_root, "stats_main.mat"));
load(fullfile(ds_root, "stats_holdout.mat"));
load(fullfile(ds_root, "stats_long.mat"));

func = @(vec) [mean(vec), std(vec)];

% cross-sectional data
tbl = [stats_main; stats_holdout];

% tbl = stats_holdout;

stats_struct(1) = table2struct(stats_main, "ToScalar", 1);
stats_struct(2) = table2struct(stats_holdout, "ToScalar", 1);
stats_struct(3) = table2struct(stats_long, "ToScalar", 1);

%% histograms

close all
fg = figure;

layout = tiledlayout(fg, 3, 3);
locs = [1 2 3 4 5 6 7];
rows = ones(1, 7);
cols = ones(1, 7);
rc = [rows(:), cols(:)];
for k = 1:numel(locs)
    ax(k) = nexttile(layout, locs(k), rc(k,:));
    hold on
end

plot_histograms([tbl.dbp, tbl.sbp], 50, ax(1));
plot_histograms((tbl.sbp - tbl.dbp), 50, ax(2));

ax(1).XAxis.Label.String = "SBP, DBP (mm Hg)";
ax(2).XAxis.Label.String = "PP (mm Hg)";

smin = tbl.zmin*1e3;
smax = tbl.zmax*1e3;

mask1 = smin >= prctile(smin, 2.5);
mask2 = smax <= prctile(smax, 97.5);

% smin = smin(mask1 & mask2);
% smax = smax(mask1 & mask2);

plot_histograms([smin, smax], -50:50, ax(3));
plot_histograms((smax - smin), 0:50, ax(4));

ax(3).XAxis.Label.String = "zMax, zMin (mOhms)";
ax(4).XAxis.Label.String = "P2P resistance (mOhms)";

smin = tbl.cmin*1e3;
smax = tbl.cmax*1e3;

mask1 = smin >= prctile(smin, 2.5);
mask2 = smax <= prctile(smax, 97.5);

% smin = smin(mask1 & mask2);
% smax = smax(mask1 & mask2);

plot_histograms([smin, smax], -50:50, ax(5));
plot_histograms((smax - smin), 0:50, ax(6));

ax(5).XAxis.Label.String = "cMax, cMin (mS/m)";
ax(6).XAxis.Label.String = "P2P conductivity (mS/m)";

plot_histograms(tbl.hr, 50, ax(7));
ax(7).XAxis.Label.String = "HR (bpm)";

fgName = "features_cross";
fgPath = fullfile(ds_root, fgName);

% exportgraphics(fg, fgPath + ".pdf", "ContentType", "vector", "Resolution", 600);


%% plot box-whisker

close all
fg = figure;

layout = tiledlayout(fg, 2, 3);
locs = 1:6;
rows = ones(1, 6);
cols = ones(1, 6);
rc = [rows(:), cols(:)];
for kk = 1:numel(locs)
    ax(kk) = nexttile(layout, locs(kk), rc(kk,:));
    hold on

    ax(kk).XAxis.Visible = 0;
    ax(kk).Box = 0;
    
end

for k = 1:3
    sbp = stats_struct(k).sbp;
    dbp = stats_struct(k).dbp;
    delta = sbp - dbp;

    plot_tukey(sbp, k, [], ax(1));
    plot_tukey(dbp, k, [], ax(2));
    plot_tukey(delta, k, [], ax(3));
end

ax(1).YLabel.String = "sbp (mm Hg)";
ax(1).YLim = [30 200];
ax(1).YTick = sort([ax(1).YLim, mean(ax(1).YLim)]);

ax(2).YLabel.String = "dbp (mm Hg)";
ax(2).YLim = ax(1).YLim;
ax(2).YTick = ax(1).YTick;

ax(3).YLabel.String = "pp (mm Hg)";
ax(3).YLim = [0 130];
ax(3).YTick = sort([ax(3).YLim, mean(ax(3).YLim)]);

for k = 1:3
    hr = stats_struct(k).hr;
    plot_tukey(hr, k, [], ax(4));
end

ax(4).YLabel.String = "hr (bpm)";
ax(4).YLim = [20 130];
ax(4).YTick = sort([ax(4).YLim, mean(ax(4).YLim)]);

for k = 1:3
    smax = stats_struct(k).zmax*1e3;
    smin = stats_struct(k).zmin*1e3;

    delta = smax - smin;
    mask1 = (delta >= 0);
    mask2 = (delta <= 40);
    delta = delta(mask1 & mask2);

    plot_tukey(delta, k, 99, ax(5));
end

ax(5).YLabel.String = "deltaZ (mOhm)";
ax(5).YLim = [0 50];
ax(5).YTick = sort([ax(5).YLim, mean(ax(5).YLim)]);

for k = 1:3
    smax = stats_struct(k).cmax*1e3;
    smin = stats_struct(k).cmin*1e3;

    delta = smax - smin;
    mask1 = (delta >= 0);
    mask2 = (delta <= 60);
    delta = delta(mask1 & mask2);

    plot_tukey(delta, k, 99, ax(6));
end

ax(6).YLabel.String = "deltaS (mS/m)";
ax(6).YLim = [0 70];
ax(6).YTick = sort([ax(6).YLim, mean(ax(6).YLim)]);

fgName = "features_tukeys";
fgPath = fullfile(ds_root, fgName);

exportgraphics(fg, fgPath + ".pdf", "ContentType", "vector", "Resolution", 600);

%% plot violin

close all
fg = figure;

layout = tiledlayout(fg, 2, 3);
locs = 1:6;
rows = ones(1, 6);
cols = ones(1, 6);
rc = [rows(:), cols(:)];
for kk = 1:numel(locs)
    ax(kk) = nexttile(layout, locs(kk), rc(kk,:));
    hold on

    ax(kk).XAxis.Visible = 0;
    ax(kk).Box = 0;
    
end

for k = 1:3
    sbp = stats_struct(k).sbp;
    dbp = stats_struct(k).dbp;
    delta = sbp - dbp;

    plot_violin(sbp, k, [], ax(1));
    plot_violin(dbp, k, [], ax(2));
    plot_violin(delta, k, [], ax(3));
end

ax(1).YLabel.String = "sbp (mm Hg)";
ax(1).YLim = [30 200];
ax(1).YTick = sort([ax(1).YLim, mean(ax(1).YLim)]);

ax(2).YLabel.String = "dbp (mm Hg)";
ax(2).YLim = ax(1).YLim;
ax(2).YTick = ax(1).YTick;

ax(3).YLabel.String = "pp (mm Hg)";
ax(3).YLim = [0 130];
ax(3).YTick = sort([ax(3).YLim, mean(ax(3).YLim)]);

for k = 1:3
    hr = stats_struct(k).hr;
    plot_violin(hr, k, [], ax(4));
end

ax(4).YLabel.String = "hr (bpm)";
ax(4).YLim = [20 130];
ax(4).YTick = sort([ax(4).YLim, mean(ax(4).YLim)]);

for k = 1:3
    smax = stats_struct(k).zmax*1e3;
    smin = stats_struct(k).zmin*1e3;

    delta = smax - smin;
    mask1 = (delta >= 0);
    mask2 = (delta <= 40);
    delta = delta(mask1 & mask2);

    plot_violin(delta, k, 99, ax(5));
end

ax(5).YLabel.String = "deltaZ (mOhm)";
ax(5).YLim = [0 50];
ax(5).YTick = sort([ax(5).YLim, mean(ax(5).YLim)]);

for k = 1:3
    smax = stats_struct(k).cmax*1e3;
    smin = stats_struct(k).cmin*1e3;

    delta = smax - smin;
    mask1 = (delta >= 0);
    mask2 = (delta <= 60);
    delta = delta(mask1 & mask2);

    plot_violin(delta, k, 99, ax(6));
end

ax(6).YLabel.String = "deltaS (mS/m)";
ax(6).YLim = [0 70];
ax(6).YTick = sort([ax(6).YLim, mean(ax(6).YLim)]);

fgName = "features_violins";
fgPath = fullfile(ds_root, fgName);

exportgraphics(fg, fgPath + ".pdf", "ContentType", "vector", "Resolution", 600);