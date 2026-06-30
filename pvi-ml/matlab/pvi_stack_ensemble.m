clear
close all

ds_root = "D:\PviProject\datasets";

ds_subdir = fullfile(ds_root,"longitudinal");

interps_dir = fullfile(ds_subdir,"_interps");
interps_subject = fullfile(interps_dir,"subjects");

files_bp = dir(fullfile(interps_subject,"*bp.csv"));

matName = "bp_interps.mat";
matPath = fullfile(ds_root, matName);

files_pviHP = dir(fullfile(interps_subject,"*pviHP.csv"));
files_pviLP = dir(fullfile(interps_subject,"*pviLP.csv"));

%% preallocate
num_samples = 0;
for k = 1:numel(files_bp)
    file = files_bp(k);
    fPath = fullfile(file.folder, file.name);
    bp = readmatrix(fPath);
    num_rows = size(bp, 1);
    num_samples = num_samples + num_rows;
end

%% make pvi data
T = 50;
bp_data = zeros(num_samples, T);

row_start = 0;
row_end = 0;
for k = 1:numel(files_bp)
    file = files_bp(k);
    fPath = fullfile(file.folder, file.name);
    
    M = readmatrix(fPath);
    num_rows = size(M, 1);
    row_start = row_end+1;
    row_end = row_end+num_rows;
    rows = row_start:row_end;

    bp_data(rows,:) = M;
end

% bp_long = bp_data;
% save(matPath, "bp_long",'-append');

%% plotting