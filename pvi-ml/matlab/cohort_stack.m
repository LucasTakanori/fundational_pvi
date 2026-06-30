clear
close all

ds_root = "D:\PviProject\datasets";

ds_subdir = fullfile(ds_root,"holdout/");

interps_dir = fullfile(ds_subdir,"_stats");
interps_subject = fullfile(interps_dir,"subjects");

stat_files = dir(fullfile(interps_subject,"*stats.csv"));

matName = "stats_holdout.mat";
matPath = fullfile(ds_root, matName);

%% preallocate
num_samples = 0;
for k = 1:numel(stat_files)
    file = stat_files(k);
    fPath = fullfile(file.folder, file.name);
    tbl = readtable(fPath);
    num_rows = size(tbl, 1);
    num_samples = num_samples + num_rows;
end

varnames = string(tbl.Properties.VariableNames);
%% make data
num_cols = size(tbl,2);
stats_data = zeros(num_samples, num_cols);
stats_data = array2table(stats_data,"VariableNames", varnames);

row_start = 0;
row_end = 0;
for k = 1:numel(stat_files)
    file = stat_files(k);
    fPath = fullfile(file.folder, file.name);

    tbl = readmatrix(fPath);
    num_rows = size(tbl, 1);
    row_start = row_end+1;
    row_end = row_end+num_rows;
    rows = row_start:row_end;

    stats_data{rows,:} = tbl;
end

stats_holdout = stats_data;
save(matPath, "stats_holdout");

func = @(vec) [mean(vec), std(vec)];