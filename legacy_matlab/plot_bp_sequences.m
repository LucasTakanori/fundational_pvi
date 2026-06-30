function fg = plot_bp_sequences(paths)
    arguments
        paths (1,1) struct
    end
    
    model_name = paths.model_name;
    h5_path = paths.h5;
    p_path = paths.partition;
    r_path = paths.results;

    subject = h5read(h5_path,'/metadata/subject');
    session = h5read(h5_path,'/metadata/session');
    num_periods = h5read(h5_path,'/metadata/num_periods');
    period_length = h5read(h5_path,'/metadata/period_length');
    num_channels = 32;

    fnames = ["train", "test", "overlaps", "uncleaned"];
    % snames = ["bp", "zHP", "zLP"];

    %% processing PARTITION TABLE
    ptbl = readtable(p_path);
    ptbl.source_name = string(ptbl.source_name);
    ptbl.partition_subset = string(ptbl.partition_subset);

    % convert python index to MATLAB index
    ptbl.gm_start = ptbl.gm_start + 1;
    ptbl.lm_start = ptbl.lm_start + 1;

    mask_session = contains(ptbl.source_name, session);
    mask_train = strcmpi(ptbl.partition_subset,'train');
    mask_test = strcmpi(ptbl.partition_subset,'test');
    mask_none = strcmpi(ptbl.partition_subset,'none');

    periods = struct();
    periods.train = ptbl.lm_end(mask_session & mask_train)';
    periods.test = ptbl.lm_end(mask_session & mask_test)';
    periods.overlaps = ptbl.lm_end(mask_session & mask_none)';
    periods.uncleaned = setdiff([1:num_periods],ptbl.lm_end(mask_session)');

    %% processing RESULTS TABLE
    rtbl = readtable(r_path); % also include results from baseline and pressor
    varnames = string(rtbl.Properties.VariableNames);

    M = table2array(rtbl);

    cols_pred = contains(varnames,"pred");
    cols_target = contains(varnames,"target");
    %
    % tmp1 = find(mask_valsalva & mask_test);
    % tmp2 = find(mask_test);
    % [~, rows] = ismember(tmp1, tmp2);

    rows = (1:height(M))';
    p2r = dictionary(ptbl.lm_end(mask_test), rows); % period to row

    %% reading hdf5

    tensors = struct();
    tensors.bp = h5read(h5_path,'/data/bp/signal');
    tensors.zHP = h5read(h5_path,'/data/pviHP/resistance');
    tensors.zLP = h5read(h5_path,'/data/pviLP/resistance');

    ds = struct();
    ds.bp = zeros(period_length, 1, num_periods);
    ds.zHP = zeros(period_length, num_channels, num_periods);
    ds.zLP = zeros(period_length, num_channels, num_periods);

    for sn = ["bp", "zHP", "zLP"]
        for p = 1:num_periods
            slice = (1:period_length)' + (p-1)*period_length;
            ds.(sn)(:,:,p) = tensors.(sn)(slice,:);
        end
    end

    ds.resistance = ds.zHP + ds.zLP;

    ds = rmfield(ds,["zHP", "zLP"]);
    %% plotting

    lw = 1.25;

    fg = figure;
    all_axes = make_tiles(fg, 3, 1);

    ax = struct();
    ax.resistance = all_axes(1);
    ax.bp = all_axes(2);
    ax.zoom = all_axes(3);

    colors.train = [hex2rgb("#3ABFC0") 1.0];
    colors.test = [hex2rgb("#FFB81D") 1.0];
    colors.uncleaned = [hex2rgb("#BE0000") 0.2];
    colors.overlaps = [hex2rgb("#707271") 0.2];


    for sn = ["resistance", "bp"]
        ax_target = ax.(sn);

        data = mean(ds.(sn),2);
        plts_lgd = [];
        for fn = fnames
            mask = periods.(fn);

            for p = mask
                tvec = (1:period_length)' + (p-1)*period_length;
                signal = data(:,:, p);

                plt = plot(ax_target, tvec, signal, ...
                    'LineWidth', lw,...
                    'Color', colors.(fn));

                if (strcmpi(sn, "bp") && strcmpi(fn, "test"))
                    row = p2r(p);
                    signal = M(row, cols_pred);
                    plt2 = plot(ax_target, tvec, signal,...
                        'LineWidth', lw,...
                        'Color', 'k', ...
                        'LineStyle', '-',...
                        'DisplayName','prediction');
                end
            end

            plt.DisplayName = fn;
            plts_lgd = [plts_lgd, plt];
        end

        ax_target.XLabel.String = 'Time';
        ax_target.XTick = [];
        ax_target.YLabel.String = sn;

        drawnow;

    end

    plts_lgd = [plts_lgd, plt2];
    lgd = legend(plts_lgd,'location','eastoutside');

    objs = findobj(ax.bp,'type','Line');
    new = copyobj(objs, ax.zoom);

    ax.zoom.XLabel.String = ax.bp.XLabel.String;
    ax.zoom.XTick = [];
    ax.zoom.YLabel.String = ax.bp.YLabel.String;

    linkaxes([ax.bp, ax.resistance], 'x');

    % find cluster
    intervals = [];
    for i = 1:numel(periods.test)
        for j = flip((i+1):numel(periods.test))
            pI = periods.test(i);
            pJ = periods.test(j);
            vec = [pI, pJ, pJ-pI];
            intervals = [intervals; vec];
        end
    end

    rows = find(intervals(:,end) <= 10);
    if ~isempty(rows)
        intervals = intervals(rows, :);
        [~, rowMax] = max(intervals(:,end));

        pI = intervals(rowMax, 1);
        pJ = intervals(rowMax, 2);

        func = @(p) (1:period_length)' + (p-1)*period_length;
        tI = func(pI);
        tJ = func(pJ);
        tMax = func(num_periods);

        tI = tI(1);
        tJ = tJ(end);

        if strcmpi(sn, "bp")
            xregion(ax.(sn), tI, tJ, ...
                'DisplayName','zoom region');
        end

        ax.zoom.XLim = [tI, tJ] + period_length*[-1 1];
    end
end