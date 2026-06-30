close all
clear all

rootdir = "D:\PviProject";
dir_artifacts = fullfile(rootdir,"artifacts","_final_ablations");

branch = "main";

all_backups = dir(dir_artifacts);
subdirs = string({all_backups([all_backups.isdir]).name});
subdirs = subdirs(~ismember(subdirs, {'.', '..'}));

% ismatch = @(s) contains(s, '-to-');
% mlDirs = subdirs(arrayfun(ismatch, subdirs));

mlDirs = subdirs;

WRITE_TBLS = 1;
WRITE_FGS = 1;

%% process and export results metrics

clear mlrp

for k = 1:numel(mlDirs)
    dir_artifact = fullfile(dir_artifacts, mlDirs(k));

    mlrp(k) = MLTRAININGREPORT(dir_artifact, branch);
    mlrp(k).process_reports();
    mlrp(k).stack_reports();
    mlrp(k).aggregate_reports();
end

matName = join(["mlrp", mlrp(1).branch],"_");
matPath = fullfile(dir_artifacts, matName + ".mat");
save(matPath, "mlrp");

%% write individual reports

for k = 1:numel(mlDirs)
    dir_artifact = fullfile(dir_artifacts, mlDirs(k));

    tblName = join(["tbl","results",mlrp(k).branch, mlrp(k).id],"_") + ".xlsx";
    tblPath = fullfile(dir_artifact, tblName);
    writetable(mlrp(k).metrics_combined, tblPath, "Sheet", "combined", "WriteRowNames", true);
    writetable(mlrp(k).metrics, tblPath,"Sheet", "single", "WriteRowNames", true);

    % tblName = join(["tbl","stats",mlrp(k).branch, mlrp(k).id],"_") + ".xlsx";
    % tblPath = fullfile(dir_artifact, tblName);
    % writetable(mlrp(k).stats, tblPath, "WriteRowNames", true);
end


%% compiling master metrics spreadsheet for all sessions

num_rows = numel(mlrp);
num_cols = size(mlrp(1).metrics_combined, 2);

A = zeros(num_rows, num_cols);
W = zeros(num_rows, num_cols);
for k = 1:numel(mlrp)
    tbl = mlrp(k).metrics_combined;

    idx_agg = find(strcmpi(tbl.Properties.RowNames,"aggregated"));
    idx_weighted = find(strcmpi(tbl.Properties.RowNames,"weighted"));

    A(k,:) = tbl{idx_agg,:};
    W(k,:) = tbl{idx_weighted,:};
end

varnames = mlrp(1).metrics_combined.Properties.VariableNames;
rownames = string([mlrp.session]);

tbl_agg = array2table(A, "VariableNames", varnames, "RowNames", rownames(:));
tbl_weighted = array2table(W, "VariableNames", varnames, "RowNames", rownames(:));

tbl_agg.Properties.DimensionNames{1} = 'session';
tbl_weighted.Properties.DimensionNames{1} = 'session';

%% write master results table
tblName_global = join(["tbl","results",mlrp(1).branch],"_") + ".xlsx";
tblPath_global = fullfile(dir_artifacts, tblName_global);
if exist(tblPath_global,'file')
    delete(tblPath_global);
end
writetable(tbl_agg, tblPath_global, "Sheet", "aggregated", "WriteRowNames", true);
writetable(tbl_weighted, tblPath_global,"Sheet", "weighted", "WriteRowNames", true);

for k = 1:numel(mlrp) % appending session tables
    tbl = mlrp(k).metrics;
    tbl.Properties.DimensionNames{1} = char(mlrp(k).session);
    writetable(tbl, tblPath_global, "Sheet", mlrp(k).id, "WriteRowNames", true);
end

%% write master stats table

% tblName_global = join(["tbl","stats",mlrp(1).branch,],"_") + ".xlsx";
% tblPath_global = fullfile(dir_artifacts, tblName_global);
% if exist(tblPath_global,'file')
%     delete(tblPath_global);
% end
% 
% for k = 1:numel(mlrp) % appending session tables
%     tbl = mlrp(k).stats; 
%     tbl.Properties.DimensionNames{1} = char(mlrp(k).session);
%     writetable(tbl, tblPath_global, "Sheet", mlrp(k).id, "WriteRowNames", true);
% end
