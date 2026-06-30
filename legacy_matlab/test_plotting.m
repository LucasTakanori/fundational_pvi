clear
close all

m = cumsum([0 31 28 31 30 31 30 31 31 30 31 30 31]);
month_labels = string([0:12]);
num_days = m(end);
t = 1:num_days;

v = 20+ 0.0005*t.^2 + 0.2*t + randn(size(t));

fg = figure;
ax = axes;
hold on
for k = 1:num_days
    x = [t(k) t(k)];
    y = [0 v(k)];

    plt = plot(x,y,'-k');

    if ismember(k, m)
        plt.LineWidth = 1;
    else
        plt.LineWidth = 0.5;
    end
end

ax.XTick = m;
ax.XTickLabel = month_labels;

axis equal
axis image

phi = linspace(0, -2*pi, num_days);
phi = phi + pi/2;

[X, Y] = pol2cart(phi, v);

fg = figure;
ax = axes;
hold on
for k = 1:num_days
    x = [0 X(k)];
    y = [0 Y(k)];

    plt = plot(x,y,'-k');

    if ismember(k, m)
        plt.LineWidth = 1;
    else
        plt.LineWidth = 0.5;
    end
end

axis equal
axis image