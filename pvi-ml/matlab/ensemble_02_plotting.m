close all
clear

rootdir = "D:\PviProject";
csvDir = fullfile(rootdir,"datasets","_csv");

format_lims = @(interval) [floor(interval(1)) ceil(interval(2))];
format_ticks = @(interval) sort([interval, mean(interval)]);

fmt = @(x) sprintf("mean:%.2f\nsd:%.2f",mean(x),std(x));

%% read master tables

files_interps = dir(fullfile(csvDir,"*interps.xlsx"));
files_stats = dir(fullfile(csvDir,"*stats.xlsx"));

files_interps = files_interps(:)';
files_stats = files_stats(:)';

signal_names = ["bp", "pviHP", "pviLP"];

for sn = signal_names
    interps.(sn) = [];
    stats.(sn) = [];
end

interps_path = fullfile(csvDir, "all_interps.xlsx");
stats_path = fullfile(csvDir, "all_stats.xlsx");

for sn = signal_names
    interps.(sn) = readtable(interps_path, Sheet=sn);
    stats.(sn) = readtable(stats_path, Sheet=sn);
end

%% plotting cohort data

csvPath = fullfile(csvDir,"notes_raw.xlsx");
tbl = readtable(csvPath);

in_to_cm = 2.54;
lbs_to_kg = 0.453592;

age = double(tbl.age);
weight = tbl.weight_lbs*lbs_to_kg;
height = tbl.height_in*in_to_cm;

cohort.sex = string(tbl.sex);
cohort.age = age(~isnan(age));
cohort.height = height(~isnan(height));
cohort.weight = weight(~isnan(weight));
cohort.bmi = cohort.weight./(cohort.height/100).^2;

nbins = 10;

close all
fg = figure;
fg.Tag = 'data_cohort';

fnames = ["age", "height", "weight", "bmi"];

ax = make_tiles(fg, 2,2);

for k = 1:numel(fnames)
    fn = fnames(k);
    histogram(ax(k), cohort.(fn), nbins);
    xline(ax(k), mean(cohort.(fn)),'-', ...
        fmt(cohort.(fn)), ...
        'LabelOrientation','horizontal');

    ax(k).XAxis.Label.String = fn;

    ax(k).TickDir = "out";
    ax(k).Box = 0;
    ax(k).PlotBoxAspectRatio = [2 1 1];

    ax(k).YLim = format_lims(ax(k).YLim);
    ax(k).YAxis.TickValues = format_ticks(ax(k).YLim);

    ax(k).XLim = format_lims(ax(k).XLim);
    ax(k).XAxis.TickValues = format_ticks(ax(k).XLim);

end

fgPath = fullfile(csvDir,fg.Tag);
exportgraphics(fg, fgPath + ".png", "BackgroundColor","none", "ContentType", "image", "Resolution", 300);
exportgraphics(fg, fgPath + ".pdf", "BackgroundColor","white", "ContentType", "vector", "Resolution", 300);

%% plotting histogram

close all

default_mask = stats.bp.mask;
pp_mask = stats.bp.height <= 100;
tmax_mask = stats.bp.tMax_rel <= 0.5;
% idx_mask = true(size(stats.bp.mask));

idx_mask = logical(default_mask .* pp_mask);

dbp = stats.bp.min(idx_mask);
sbp = stats.bp.max(idx_mask);
hr = stats.bp.hr(idx_mask);

height_bp = stats.bp.height(idx_mask);
height_pvi = 1e3*stats.pviHP.height(idx_mask);

fg = figure;
fg.Tag = "data_histogram";
nbins = 50;

ax = make_tiles(fg, 2,2);

histogram(ax(1), dbp, nbins);
histogram(ax(1), sbp, nbins);
xline(ax(1), mean(dbp),'-', fmt(dbp));
xline(ax(1), mean(sbp),'-', fmt(sbp));
ax(1).XLim = [min(dbp), max(sbp)];
ax(1).XAxis.Label.String = 'dbp, sbp';

histogram(ax(2), hr, nbins);
xline(ax(2), mean(hr),'-', fmt(hr));
ax(2).XLim = [min(hr), max(hr)];
ax(2).XAxis.Label.String = 'hr';

histogram(ax(3), height_bp, nbins);
xline(ax(3), mean(height_bp),'-', fmt(height_bp));
ax(3).XLim = [10, max(height_bp)];
ax(3).XAxis.Label.String = 'delta bp';

histogram(ax(4), height_pvi, nbins);
xline(ax(4), mean(height_pvi),'-', fmt(height_pvi));
ax(4).XLim = [min(height_pvi), max(height_pvi)];
ax(4).XAxis.Label.String = 'delta pvi';

xlines = findall(fg,'Type','ConstantLine');
for k = 1:numel(xlines)
    xlines(k).LabelOrientation = 'horizontal';
end

for k = 1:numel(ax)
    ax(k).TickDir = "out";
    ax(k).Box = 0;
    % ax(k).PlotBoxAspectRatio = [1 1 1];
    ax(k).YLim = format_lims(ax(k).YLim);
    ax(k).YAxis.TickValues = format_ticks(ax(k).YLim);
    ax(k).YAxis.TickLabels = string(ax(k).YAxis.TickValues);

    ax(k).XLim = format_lims(ax(k).XLim);
    ax(k).XAxis.TickValues = format_ticks(ax(k).XLim);
end

fgPath = fullfile(csvDir,fg.Tag);
exportgraphics(fg, fgPath + ".png", "BackgroundColor","none", "ContentType", "image", "Resolution", 300);
exportgraphics(fg, fgPath + ".pdf", "BackgroundColor","white", "ContentType", "vector", "Resolution", 300);

%% Plotting ensemble

periods = structfun(@(arr) arr{idx_mask,1:50}, interps, 'UniformOutput', false);
periods.pvi = periods.pviHP + periods.pviLP;

fnames = ["bp", "pviHP"];
factors = [1, 1e3];

close all
fg = figure;
fg.Tag = 'data_ensemble';
% fg.Tag = 'data_ensemble_average';

ax = make_tiles(fg, 2, 1);

tvec = linspace(0,1,50);

for k = 1:numel(fnames)
    fn = fnames(k);

    s = periods.(fn)*factors(k);

    sAVG = mean(s, 1);
    sDEV = std(s,[], 1);
    kDEV = 1;

    sLOW = sAVG - kDEV*sDEV;
    sHIGH = sAVG + kDEV*sDEV;

    plot(ax(k), ...
        tvec,s',...
        'LineStyle','-', ...
        'LineWidth',0.5, ...
        'Color',[0.85*[1 1 1], 0.1]);

    % plot(ax(k), ...
    %     tvec, sAVG, '-r',...
    %     'LineWidth', 1);
    % 
    % plot(ax(k), ...
    %     tvec, sHIGH, '-k',...
    %     'LineWidth', 1);
    % 
    % plot(ax(k), ...
    %     tvec, sLOW, '-k',...
    %     'LineWidth', 1);

    ax(k).YLim = [min(s(:)), max(s(:))];

end

for k = 1:numel(ax)
    ax(k).TickDir = "out";
    ax(k).Box = 0;
    ax(k).PlotBoxAspectRatio = [1.5 1 1];

    ax(k).YLim = format_lims(ax(k).YLim);
    ax(k).YAxis.TickValues = format_ticks(ax(k).YLim);

    ax(k).XLim = format_lims(ax(k).XLim);
    ax(k).XAxis.TickValues = format_ticks(ax(k).XLim);
end

fgPath = fullfile(csvDir,fg.Tag);
% exportgraphics(fg, fgPath + ".png", "BackgroundColor","none", "ContentType", "image", "Resolution", 300);
% exportgraphics(fg, fgPath + ".pdf", "BackgroundColor","white", "ContentType", "vector", "Resolution", 300);