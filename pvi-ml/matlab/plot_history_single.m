function plot_history_single(mlrp_struct, fg)
    arguments
        mlrp_struct struct
        fg (1,1) matlab.ui.Figure = gcf
    end

    clf(fg);
    
    layout = tiledlayout(2,1);

    layout.Padding = "tight";
    layout.TileSpacing = "tight";

    layout.Title.String = mlrp_struct.session;
    layout.Title.FontWeight = 'bold';

    ax(1) = nexttile(layout); hold on
    ax(2) = nexttile(layout); hold on

    mlrp_struct.reports(2:end) = [];
    num_reports = numel(mlrp_struct.reports);
    
    % cmap = lines(num_reports);
    plts = struct();
    plts.loss = struct();
    plts.bp_accuracy = struct();
    
    xrange = 0;
    yrange = 0;
    xmax = 0;
    for kk = 1:num_reports
        history_tbl = mlrp_struct.reports(kk).history;

        % tmp = movmean(diff(history_tbl.test_loss), 10);
        plts.loss.train(kk) = plot(ax(1), history_tbl.epoch, history_tbl.train_loss);
        plts.loss.test(kk) = plot(ax(1), history_tbl.epoch, history_tbl.test_loss);
        plts.bp_accuracy.train(kk) = plot(ax(2), history_tbl.epoch, history_tbl.train_accuracy);
        plts.bp_accuracy.test(kk) = plot(ax(2), history_tbl.epoch, history_tbl.test_accuracy);

        % plts.loss.train(kk).LineStyle = '-';
        % plts.loss.test(kk).LineStyle = '-.';
        % plts.bp_accuracy.train(kk).LineStyle = '-';
        % plts.bp_accuracy.test(kk).LineStyle = '-.';

        % plts.loss.train(kk).Color = cmap(kk,:);
        % plts.loss.test(kk).Color = cmap(kk,:);
        % plts.bp_accuracy.train(kk).Color = cmap(kk,:);
        % plts.bp_accuracy.test(kk).Color = cmap(kk,:);

        plts.loss.train(kk).DisplayName = 'Train';
        plts.loss.test(kk).DisplayName = 'Test';
       plts.bp_accuracy.train(kk).DisplayName = 'Train';
               plts.bp_accuracy.test(kk).DisplayName = 'Test';

        lgd(1) = legend(ax(1),'Location','northeast', 'AutoUpdate', 0);
        lgd(2) = legend(ax(2),'Location','southeast', 'AutoUpdate', 0);

        xrange = unique([xrange; history_tbl.epoch(:)]);
        yrange = unique([yrange; history_tbl.test_loss(:); history_tbl.train_loss(:)]);
        xmax = max([xmax; history_tbl.epoch(:)]);
    end
    
    yline(ax(2), 0.5, '-', 'Tracking threshold');

    % ax(1).XAxis.Limits = [0 min(500, max(history_tbl.epoch))];
    % ax(2).XAxis.Limits = [0 min(500, max(history_tbl.epoch))];

    % ax(1).YScale = 'log';
    ax(2).XAxis.Label.String = "Epoch";

    ax(1).YAxis.Label.String = "Model loss";

    xrange = [0 500];
    ax(1) = fix_axis(ax(1), xrange, ceil(yrange));
    ax(2) = fix_axis(ax(2), xrange, []);
    
    ax(2).YAxis.Label.String = "Model bp_accuracy (Pearson)";
    ax(2).YAxis.Limits = [-1 1];
    ax(2).YAxis.TickValues = -1:0.5:1;

    for k = 1:numel(ax)
        ax(k).Box = "off";
        ax(k).TickDir = "out";
        ax(k).PlotBoxAspectRatio = [2 1 1];
    end

end