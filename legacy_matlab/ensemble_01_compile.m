close all
clear

rootdir = "D:\PviProject";
csvDir = fullfile(rootdir,"datasets","_csv");

files_interps = dir(fullfile(csvDir,"_subjects", "*interps.xlsx"));
files_stats = dir(fullfile(csvDir,"_subjects", "*stats.xlsx"));

files_interps = files_interps(:)';
files_stats = files_stats(:)';

%% compiling interp and stat tables

signal_names = ["bp", "pviHP", "pviLP"];
signal_names = signal_names(:)';

for sn = signal_names
    interps.(sn) = [];
    stats.(sn) = [];
end

for file = files_interps
    filePath = fullfile(file.folder, file.name);
    for sn = signal_names
        tbl = readtable(filePath, Sheet=sn);
        interps.(sn) = vertcat(interps.(sn), tbl);
    end
end

for file = files_stats
    filePath = fullfile(file.folder, file.name);
    for sn = signal_names
        tbl = readtable(filePath, Sheet=sn);
        stats.(sn) = vertcat(stats.(sn), tbl);
    end
end

%% export master tables

% interps_path = fullfile(csvDir, "all_interps.xlsx");
% stats_path = fullfile(csvDir, "all_stats.xlsx");
% 
% for sn = signal_names
%     writetable(interps.(sn), interps_path, Sheet=sn);
%     writetable(stats.(sn), stats_path, Sheet=sn);
% end

%% plotting histogram

close all

default_mask = stats.bp.mask;
pp_mask = stats.bp.height <= 100;
% idx_mask = true(size(stats.bp.mask));

idx_mask = logical(default_mask .* pp_mask);

dbp = stats.bp.min(idx_mask);
sbp = stats.bp.max(idx_mask);
hr = stats.bp.hr(idx_mask);

height_bp = stats.bp.height(idx_mask);
height_pvi = 1e3*stats.pviHP.height(idx_mask);

fmt = @(x) sprintf("mean:%.2f\nsd:%.2f",mean(x),std(x));

fg = figure;

fnames = ["bp", "pviHP"];

nbins = 50;

layout = tiledlayout(fg, 2, 2);
locs = [1 2 3 4];
rows = ones(1, 4);
cols = ones(1, 4);
rc = [rows(:), cols(:)];
for k = 1:numel(locs)
    ax(k) = nexttile(layout, locs(k), rc(k,:));
    hold on
end

% layout.Padding = "tight";
layout.TileSpacing = "tight";

histogram(ax(1), dbp, nbins);
histogram(ax(1), sbp, nbins);
xline(ax(1), mean(dbp),'-', fmt(dbp));
xline(ax(1), mean(sbp),'-', fmt(sbp));
ax(1).XLim = [min(dbp), max(sbp)];
ax(1).Title.String = 'dbp, sbp';

histogram(ax(2), hr, nbins);
xline(ax(2), mean(hr),'-', fmt(hr));
ax(2).XLim = [min(hr), max(hr)];
ax(2).Title.String = 'hr';

histogram(ax(3), height_bp, nbins);
xline(ax(3), mean(height_bp),'-', fmt(height_bp));
ax(3).XLim = [10, max(height_bp)];
ax(3).Title.String = 'delta bp';

histogram(ax(4), height_pvi, nbins);
xline(ax(4), mean(height_pvi),'-', fmt(height_pvi));
ax(4).XLim = [min(height_pvi), max(height_pvi)];
ax(4).Title.String = 'height pvi';

xlines = findall(fg,'Type','ConstantLine');
for k = 1:numel(xlines)
    xlines(k).LabelOrientation = 'horizontal';
end

for k = 1:numel(ax)
    ax(k).TickDir = "out";
    ax(k).Box = 0;
    % ax(k).PlotBoxAspectRatio = [1 1 1];

    yTicks = sort([ax(k).YLim mean(ax(k).YLim)]);
    % ax(k).YAxis.TickValues = yTicks;

    ax(k).XLim = [floor(ax(k).XLim(1)), ceil(ax(k).XLim(2))];
    xTicks = sort([ax(k).XLim mean(ax(k).XLim)]);
    % xTicks = round(xTicks*100)/100;
    ax(k).XAxis.TickValues = xTicks;
end

%% Plotting ensemble

periods = structfun(@(arr) arr{idx_mask,1:50}, interps, 'UniformOutput', false);

periods.pvi = periods.pviHP + periods.pviLP;
layout.TileSpacing = "tight";

fnames = ["bp", "pvi", "pviHP", "pviLP"];

fg = figure;

layout = tiledlayout(fg, 2, 2);
locs = [1 2 3 4];
rows = ones(1, 4);
cols = ones(1, 4);
rc = [rows(:), cols(:)];
for k = numel(locs)
    ax(k) = nexttile(layout, locs(k), rc(k,:));
    hold on
end

for k = 1:numel(fnames)
    fn = fnames(k);

    s = periods.(fn);
    
    sAVG = mean(s, 1);
    sDEV = std(s,[], 1);
    kDEV = 1;

    sLOW = sAVG - kDEV*sDEV;
    sHIGH = sAVG - kDEV*sDEV;

    
end