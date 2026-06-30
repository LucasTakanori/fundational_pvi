clear

rootdir = "D:\PviProject";
dir_artifacts = fullfile(rootdir,"artifacts","_final_ps");

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

%% plotting history

close all

for k = 1:numel(mlrp) % exporting individual figures
    fg(k) = figure();
    history_tbl = mlrp(k).reports(1).history;
    plot_history_single(mlrp(k), fg(k));
    drawnow;
end

%% export

for k = 1:numel(mlrp)
    fgName = join(["fg_history", mlrp(k).id],"-");
    fgPath = fullfile(mlrp(k).dir, fgName);
    exportgraphics(fg(k), fgPath + ".png", "BackgroundColor","none", "ContentType", "image", "Resolution", 300);
    exportgraphics(fg(k), fgPath + ".pdf", "BackgroundColor","none", "ContentType", "vector", "Resolution", 300);
end


fgName_global = join(["fg_history",mlrp(1).branch,],"_");
fgPath_global = fullfile(dir_artifacts, fgName_global) + ".pdf";

if exist(fgPath_global,'file')
    delete(fgPath_global);
end

% compiling multilple figures into a single pdf
for k = 1:numel(mlrp)
    exportgraphics(fg(k), fgPath_global, "BackgroundColor","white", "ContentType", "image", "Resolution", 300, "Append",true);
end