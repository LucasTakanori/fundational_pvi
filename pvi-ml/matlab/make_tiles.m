function all_axes = make_tiles(fg, m, n)
    layout = tiledlayout(fg, m, n);
    for k = 1:(m*n)
        all_axes(k) = nexttile; hold on
        
        all_axes(k).TickDir = 'out';
        all_axes(k).Box = 'off';
    end
    
    layout.TileSpacing = "tight";
    layout.Padding = "tight";