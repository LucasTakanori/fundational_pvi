clear
close all

ds_root = "D:\PviProject\datasets";

ds_subdir = fullfile(ds_root,"holdout");

interps_dir = fullfile(ds_subdir,"_interps");
interps_subject = fullfile(interps_dir,"subjects");

files_bp = dir(fullfile(interps_subject,"*bp.csv"));

matName = "interps_holdout.mat";
matPath = fullfile(ds_root, matName);

files_pviHP = dir(fullfile(interps_subject,"*pviHP.csv"));
files_pviLP = dir(fullfile(interps_subject,"*pviLP.csv"));

T = 50;
C = 32;
nCols = T*C;
%% preallocate
offsets = [];
for k = 1:numel(files_bp)
    file = files_bp(k);
    fPath = fullfile(file.folder, file.name);
    bp = readmatrix(fPath);
    offsets = [offsets, size(bp, 1)];
end

num_samples = sum(offsets);

%% read
bp = zeros(num_samples, T);
zHP = zeros(num_samples, C*T);
zLP = zeros(num_samples, C*T);

row_start = 0;
row_end = 0;
for k = 1:numel(files_bp)
    num_rows = offsets(k);
    row_start = row_end+1;
    row_end = row_end+num_rows;
    rows = row_start:row_end;

    file = files_bp(k);
    fPath = fullfile(file.folder, file.name);
    M = readmatrix(fPath);

    bp(rows,:) = M;

    file = files_pviHP(k);
    fPath = fullfile(file.folder, file.name);
    M = readmatrix(fPath);
    M = M(:,(nCols+1):end); % extract real part

    zHP(rows,:) = M;

    file = files_pviHP(k);
    fPath = fullfile(file.folder, file.name);
    M = readmatrix(fPath);
    M = M(:,(nCols+1):end); % extract real part

    zLP(rows,:) = M;

end

% The R/X channels in the Torch tensor was swapped!!!
% Ideally, we want R first, then X, but now it's reversed
% pviHP = pviHP(:,1:nCols); % this is wrong! See below

% pviHP = pviHP(:,(nCols+1):end);

%%

tmp1 = zeros(num_samples, T, C);
tmp2 = zeros(num_samples, T, C);
for i = 1:C
    cols = (1:T) + (i-1)*T;
    tmp1(:,:,i) = zHP(:,cols);
    tmp2(:,:,i) = zLP(:,cols);
end

zHP = tmp1;
zLP = tmp2;

save(matPath, "bp", "zHP", "zLP", "-v7.3");

%%
% tmp2 = zeros(C, T*num_samples);
% for i = 1:C
%     p = tmp(:,:,i)';
%     s = reshape(p,1,[]);
%     tmp2(i,:) = s;
% end
% 
% pvi = tmp2;
% pvi_signal = mean(pvi,1);
% bp_signal = reshape(bp',1,[]);
% 
% close all
% fg = figure;
% 
% ax1 = subplot(2,1,1);
% plot(ax1,bp_signal);
% 
% ax2 = subplot(2,1,2);
% plot(ax2,pvi_signal);
% 
% linkaxes([ax1 ax2], "x");