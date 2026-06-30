clear
close all

rootdir = "D:\PviProject";
dir_artifacts = fullfile(rootdir,"artifacts","_final_ablations");

all_backups = dir(dir_artifacts);
subdirs = string({all_backups([all_backups.isdir]).name});
subdirs = subdirs(~ismember(subdirs, {'.', '..'}));

% ismatch = @(s) contains(s, '-to-');
% mlDirs = subdirs(arrayfun(ismatch, subdirs));

mlDirs = subdirs;

matPath = fullfile(dir_artifacts, "mlrp_main.mat");
load(matPath, "mlrp");

sbp = @(X) max(X,[],2);
dbp = @(X) min(X,[],2);
preds = @(X) X(:,1:size(X,2)/2);
targets = @(X) X(:,size(X,2)/2+1:end);

mae = @(y1, y2) mean(abs(y1-y2));
rmse = @(y1, y2) sqrt(mean((y1-y2).^2));
%% plotting results

close all

for k = 1:numel(mlrp) % exporting individual figures
    fg(k) = figure();
    plot_results_single(mlrp(k), fg(k));
    drawnow;
end

%% exporting figures

for k = 1:numel(mlrp)
    fgName = join(["fg","results",mlrp(k).branch, mlrp(k).id],"_");
    fgPath = fullfile(mlrp(k).dir, fgName);
    exportgraphics(fg(k), fgPath + ".png", "BackgroundColor","white", "ContentType", "image", "Resolution", 600);
    exportgraphics(fg(k), fgPath + ".pdf", "BackgroundColor","none", "ContentType", "vector", "Resolution", 600);
end

fgName_global = join(["fg","results",mlrp(1).branch],"_");
fgPath_global = fullfile(dir_artifacts, fgName_global) + ".pdf";

if exist(fgPath_global,'file')
    delete(fgPath_global);
end

% compiling multilple figures into a single pdf
for k = 1:numel(mlrp)
    exportgraphics(fg(k), fgPath_global, "BackgroundColor","white", "ContentType", "vector", "Resolution", 300, "Append",true);
end