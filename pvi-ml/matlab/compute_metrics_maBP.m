close all
clear all

rootdir = "D:\PviProject";
dir_artifacts = fullfile(rootdir,"artifacts","_final_pw");
% dir_model = fullfile(dir_artifacts, "pw15-crt-bioz-to-waveform");
dir_model = fullfile(dir_artifacts, "pw13-crt-img-to-waveform");

csvDir = fullfile(dir_model,"main","results");

csvFiles = dir(csvDir);
csvFiles = csvFiles(:)';
csvFiles = csvFiles(~[csvFiles.isdir]);

extract_mBP = @(s) 1/3*max(s, [] ,2) + 2/3*min(s, [], 2);

sBP_preds = [];
sBP_targets = [];

dBP_preds = [];
dBP_targets = [];

for file = csvFiles
    csvPath = fullfile(file.folder, file.name);
    data = table2array(readtable(csvPath));
    
    numcols = size(data,2);
    preds = data(:, 1:numcols/2);
    targets = data(:, (numcols/2 + 1):end);

    sBP_preds = [sBP_preds; max(preds, [], 2)];
    sBP_targets = [sBP_targets; max(targets, [], 2)];

    dBP_preds = [dBP_preds; min(preds, [], 2)];
    dBP_targets = [dBP_targets; min(targets, [], 2)];
end

mBP_preds = 1/3*sBP_preds + 2/3*dBP_preds;
mBP_targets = 1/3*sBP_targets + 2/3*dBP_targets;

errs = mBP_preds - mBP_targets;
 
ME = mean(errs);
SDE = std(errs);
 
ciLOW = prctile(errs, 2.5);
ciHIGH = prctile(errs, 97.5);

[ME, ciLOW, ciHIGH]