clear
close all

rootdir = "D:\PviProject";
dir_artifacts = fullfile(rootdir,"artifacts","_final_ss");

all_backups = dir(dir_artifacts);
subdirs = string({all_backups([all_backups.isdir]).name});
subdirs = subdirs(~ismember(subdirs, {'.', '..'}));

% ismatch = @(s) contains(s, '-to-');
% mlDirs = subdirs(arrayfun(ismatch, subdirs));

mlDirs = subdirs;

WRITE_TBLS = 1;
WRITE_FGS = 1;

matPath = fullfile(dir_artifacts, "mlrp_main.mat");
load(matPath, "mlrp");

%% extract

mlrpW = [];
keys = [];
for k = 1:numel(mlrp)
    session = mlrp(k).session;

    if ~contains(session,"waveform")
        continue;
    end

    mlrpW = [mlrpW; mlrp(k)];

    parts = strsplit(session ,'-');
    keys = [keys; parts(:)'];
end

% [~, idx] = sortrows(keys,[3 1]);
% keys = keys(idx,:);
% mlrpW = mlrpW(idx);

%% plotting stats
close all

fg = figure;
layout = tiledlayout(fg, 10,2);
num_axes = prod(layout.GridSize);

for k = 1:num_axes
    ax(k) = nexttile(layout);
    hold on
end

layout.Padding = "tight";
layout.TileSpacing = "tight";

yrange = [];
wrange = [];
for k = 1:numel(mlrpW)
    loc1 = 2*k-1;
    loc2 = loc1 + 1;

    s = mlrpW(k).stats.sqi;
    w = mlrpW(k).stats.w1_label;
    % x2 = x2/max(x2);

    wrange = unique([wrange; w(:)]);

    y = mlrpW(k).stats.amae;
    yrange = unique([yrange; y(:)]);

    scatter_regress(ax(loc1), s, y);
    scatter_regress(ax(loc2), w, y);

    session = mlrpW(k).session;
    parts = strsplit(session ,'-');
    kw = parts(1);

    ax(loc1).Title.String = kw;
    ax(loc2).Title.String = kw;

    % ax(k) = fix_axis(ax(k));
end

yrange = [0, round(max(yrange),-1)];
wrange = [floor(min(wrange)), ceil(max(wrange))];
for k = 1:numel(ax)
    ax(k) = fix_axis(ax(k), [], yrange);

    if ~mod(k,2)
        ax(k) = fix_axis(ax(k), wrange, yrange);
    end

    ax(k).PlotBoxAspectRatio = [4 1 1];
    ax(k).TickDir = "none";
end

fgDir = dir_artifacts;
fgName = join(["fg","robustness",mlrp(1).branch],"_");
fgPath = fullfile(fgDir, fgName) + ".pdf";
exportgraphics(fg, fgPath, "BackgroundColor","none", "ContentType", "vector", "Resolution", 300);

%% plotting epoch

close all

fg = figure;
layout = tiledlayout(fg, 10,2);
num_axes = prod(layout.GridSize);

for k = 1:num_axes
    ax(k) = nexttile(layout);
    hold on
end

layout.Padding = "tight";
layout.TileSpacing = "tight";

for k = 1:numel(mlrpW)
    loc1 = 2*k-1;
    loc2 = loc1 + 1;

    s = mlrpW(k).stats.sqi;
    w = mlrpW(k).stats.w1_domain;
    w = w/max(w);

    y = mlrpW(k).stats.amae;
    yrange = unique([yrange; y(:)]);

    scatter_regress(ax(loc1), s, y);
    scatter_regress(ax(loc2), w, y);

    session = mlrpW(k).session;
    parts = strsplit(session ,'-');
    kw = parts(1);

    ax(loc1).Title.String = kw;
    ax(loc2).Title.String = kw;

    % ax(k) = fix_axis(ax(k));
end