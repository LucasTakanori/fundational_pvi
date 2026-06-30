close all
clear all

funcs = {...
    @(n) log2(n);...
    @(n) sqrt(n);
    @(n) n;...
    @(n) n.*log2(n);...
    @(n) n.^2;...
    @(n) n.^3;...
    @(n) 2.^n;...
    @(n) factorial(n);
    };

fg = figure();
ax = axes(); hold on

n = 1:1000;
for k = 1:numel(funcs)
    fn = funcs{k};
    plot(n, fn(n));
end

ylim([1 1000]);
xlim([1 1000]);

xscale('log');
yscale('log');