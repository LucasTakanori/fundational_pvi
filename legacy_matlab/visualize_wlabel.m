clear
close all

% bp_train = readmatrix("bp_train.csv");
% bp_test = readmatrix("bp_test.csv");
% bp_holdout = readmatrix("bp_holdout.csv");

% fg = figure;
% ax = axes; hold on
% 
% sct1 = scatter(ax, bp_train(:,1), bp_train(:,2));
% sct2 = scatter(ax, bp_test(:,1), bp_test(:,2));
% sct3 = scatter(ax, bp_holdout(:,1), bp_holdout(:,2));



f = @(x, mu, var) 1/sqrt(2*pi*var).*exp(-(x-mu).^2./(2*var));

x = linspace(-20, 20, 5000);

f1 = f(x, -6, 3) + f(x, -2, 4);
f2 = f(x, 1, 2.5) + f(x, -2, 1);
f3 = f(x, 5, 2) + f(x, 9, 5);

fg = figure;
ax = axes; hold on

plot(ax, x, f1);
plot(ax, x, f2);
plot(ax, x, f3);

exportgraphics(fg, "wlabel-test.pdf", "BackgroundColor","white", "ContentType", "vector", "Resolution", 300, "Append",true);