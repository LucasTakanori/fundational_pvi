close all
clear

num_points = 500;
num_trials = 500;
colors = parula(num_trials);
alpha = 0.01;

t = linspace(0,4*pi, num_points);

xtrue = 100*(0.1*cos(2*t) + 0.15*sin(3*t) + 0.1*sin(t));
ytrue = 100*(0.3*sin(t) + 0.2*sin(1.5*t) + 0.05*sin(2*t));

X = zeros(num_trials, num_points);
Y = zeros(num_trials, num_points);

for k = 1:num_trials
    x_observed = xtrue + randn(size(xtrue));
    y_observed = ytrue + randn(size(ytrue));

    X(k,:) = x_observed;
    Y(k,:) = y_observed;
end

xmean = mean(X, 1);
ymean = mean(Y, 1);

xdev = std(X,[],1);
ydev = std(Y,[],1);

%% Ensemble plots

close all

fg = figure;
layout = tiledlayout(4,2,"TileSpacing","tight","Padding","tight");
ax1 = nexttile(layout, 1, [1 2]); hold on
ax2 = nexttile(layout, 3, [1 2]); hold on
ax3 = nexttile(layout, 5, [2 2]); hold on

ax1.PlotBoxAspectRatio = [2 1 1];
ax2.PlotBoxAspectRatio = [2 1 1];

ax3.DataAspectRatio = [1 1 1];
ax3.PlotBoxAspectRatio = [1 1 1];

for k = 1:num_trials
    x_observed = X(k,:);
    y_observed = Y(k,:);

    plt = plot(ax1, t, x_observed, '-');
    plt.Color = [0.85*[1 1 1] alpha];
   
    plt = plot(ax2, t, y_observed, '-');
    plt.Color = [0.85*[1 1 1] alpha];

    plt = plot(ax3, x_observed, y_observed, '-');
    plt.Color = [0.85*[1 1 1] alpha];
end

plt = plot(ax1, t, xtrue, '-k');
plt = plot(ax2, t, ytrue, '-k');
plt = plot(ax3, xtrue, ytrue, '-k');

%% Mean plots

close all

fg = figure;
layout = tiledlayout(4,2,"TileSpacing","tight","Padding","tight");
ax1 = nexttile(layout, 1, [1 2]); hold on
ax2 = nexttile(layout, 3, [1 2]); hold on
ax3 = nexttile(layout, 5, [2 2]); hold on

ax1.PlotBoxAspectRatio = [2 1 1];
ax2.PlotBoxAspectRatio = [2 1 1];

ax3.DataAspectRatio = [1 1 1];
ax3.PlotBoxAspectRatio = [1 1 1];

plt = plot(ax1, t, xmean + xdev, '-b');
plt = plot(ax1, t, xmean - xdev, '-b');

plt = plot(ax2, t, ymean + ydev, '-b');
plt = plot(ax2, t, ymean - ydev, '-b');

xydev = sqrt(xdev.^2 + ydev.^2);

plt = plot(ax3, xmean + xydev, ymean + xydev, '-b');
plt = plot(ax3, xmean - xydev, ymean - xydev, '-b');
plt = plot(ax3, xmean + xydev, ymean - xydev, '-b');
plt = plot(ax3, xmean - xydev, ymean + xydev, '-b');

plt = plot(ax1, t, xmean, '-r');
plt = plot(ax2, t, ymean, '-r');
plt = plot(ax3, xmean, ymean, '-r');