classdef MLTRAININGREPORT < handle
    properties (SetAccess = private)
        session
        id
        date
        dir
        branch
        reports
        results
        metrics
        stats
        metrics_combined
    end

    properties (Access = private)
        rDir % results dir
        hDir % history dir
        cDir % configs dir
        sDir % statistics dir

        hFiles
        rFiles
        cFiles
        sFiles

        rNames
        hNames
        cNames
        sNames
    end

    methods
        function obj = MLTRAININGREPORT(logdir, branch)
            arguments
                logdir string
                branch string
            end

            [~, dirName, ~] = fileparts(logdir);

            obj.session = dirName;
            obj.id = regexp(dirName,"^([^-]*)","match","once");
            obj.date = regexp(dirName, "\d{8}", "match", "once");
            obj.dir = logdir;

            fprintf("\n");
            fprintf("MLTRAININGREPORT:[Initiating MLRP class for session: '%s']\n", obj.session);
            fprintf("LOGDIR set to: '%s'\n", obj.dir);
            
            obj.branch = branch;

            obj.rDir = fullfile(obj.dir,obj.branch,"results");
            obj.hDir = fullfile(obj.dir,obj.branch,"history");
            obj.cDir = fullfile(obj.dir,obj.branch,"configs");
            obj.sDir = fullfile(obj.dir,obj.branch,"statistics");

            obj.rFiles = dir(fullfile(obj.rDir,"*results.csv"));
            obj.hFiles = dir(fullfile(obj.hDir,"*history.csv"));
            obj.cFiles = dir(fullfile(obj.cDir,"*configs.json"));
            obj.sFiles = dir(fullfile(obj.sDir,"*statistics.json"));

            idx_rmv = [];
            for k = 1:numel(obj.rFiles)
               if ~contains(obj.rFiles(k).name,"subject")
                   idx_rmv = [idx_rmv, k];
               end
            end

            obj.rFiles(idx_rmv) = [];

            % obj.rFiles = obj.rFiles(1:1);
            % obj.hFiles = obj.hFiles(1:1);

            func = @(str, kw) string(regexprep(str,"^(.*)(?:_" + kw + "?).*$", "$1"));
            obj.rNames = arrayfun(@(s) func(s.name,'results'), obj.rFiles);
            % obj.hNames = arrayfun(@(s) func(s.name,"history"), obj.hFiles);
            % obj.cNames = arrayfun(@(s) func(s.name,"configs"), obj.cFiles);
            % obj.sNames = arrayfun(@(s) func(s.name,"statistics"), obj.sFiles);

            fprintf("\t Results dir: '%s'. (Found %d results files)\n", obj.rDir, numel(obj.rFiles));
            fprintf("\t History dir: '%s'. (Found %d history files)\n", obj.hDir, numel(obj.hFiles));
            fprintf("\t Configs dir: '%s'. (Found %d configs files)\n", obj.cDir, numel(obj.cFiles));
            fprintf("\t Statistics dir: '%s'. (Found %d statistics files)\n", obj.sDir, numel(obj.sFiles));
            
            obj.populate_reports();

            if ~exist(obj.rDir, 'dir') || numel(obj.rFiles)==0
                warning("[MLTRAININGREPORT: Results not available for '%s'!]", obj.session);
            end
        end

        function populate_reports(obj)
            
            rp_single = struct();
            rp_single.dataset_name = [];
            
            rp_single.epoch = 0; % history
            
            rp_single.num_train = 0;
            rp_single.num_test = 0;
            
            rp_single.results = [];
            rp_single.metrics = [];
            rp_single.history = [];
            rp_single.stats = [];

            obj.reports = repmat(rp_single, [numel(obj.rFiles), 1]);
            for k = 1:numel(obj.rFiles)
                obj.reports(k).dataset_name = obj.rNames(k);
            end
        end

        function read_results(obj)
            for k = 1:numel(obj.rFiles)
                rPath = fullfile(obj.rFiles(k).folder, obj.rFiles(k).name);
                rp_results = readtable(rPath); % predictions and targets
                obj.reports(k).results = rp_results;
                obj.reports(k).num_test = size(rp_results, 1);
                obj.reports(k).metrics = compute_ml_metrics(rp_results);
            end
        end

        function read_history(obj)
            for k = 1:numel(obj.hFiles)
                hPath = fullfile(obj.hFiles(k).folder, obj.hFiles(k).name);

                history = readtable(hPath); % history
                history.epoch = history.epoch+1;

                obj.reports(k).epoch = height(history);
                obj.reports(k).history = history;
            end

        end

        function read_configs(obj)
            for k = 1:numel(obj.cFiles)
                cPath = fullfile(obj.cFiles(k).folder, obj.cFiles(k).name);
                configs = jsondecode(fileread(cPath));
                obj.reports(k).configs = configs;
            end
        end

        function read_stats(obj)
            for k = 1:numel(obj.sFiles)
                sPath = fullfile(obj.sFiles(k).folder, obj.sFiles(k).name);
                rp_stats = jsondecode(fileread(sPath));
                % rp_stats.epoch = rp_stats.epoch + 1;
                rp_stats = struct2table(rp_stats,'AsArray',true);
                
                obj.reports(k).num_train = rp_stats.num_train;
                % obj.reports(k).num_test = rp_stats.num_test;
                % obj.reports(k).epoch = rp_stats.epoch;
                obj.reports(k).stats = rp_stats;
            end
        end

        function process_reports(obj)
            t1 = datetime;

            obj.read_results();
            obj.read_history();
            obj.read_configs();
            obj.read_stats();

            t2 = datetime;
            fprintf("\t ...Done! (%.2f seconds)\n", seconds(t2 - t1));

        end

        function stack_statistics(obj)
            stats_out = [];
            for k = 1:numel(obj.reports)
                rp_stats = obj.reports(k).stats;
                stats_out = [stats_out; rp_stats];
            end
            stats_out.Properties.RowNames = obj.rNames;
            obj.stats = stats_out;
        end

        function stack_metrics(obj)
            metrics_out = [];
            for k = 1:numel(obj.reports)
                rp = obj.reports(k);
                
                tbl = table(rp.num_train, rp.num_test, rp.epoch, ...
                    'VariableNames', ["num_train", "num_test", "epoch"]);
                
                tbl = [tbl, rp.metrics];
                metrics_out = [metrics_out; tbl];

            end

            metrics_out.Properties.RowNames = obj.rNames;
            obj.metrics = metrics_out;

        end

        function stack_results(obj)
            M = [];
            for k = 1:numel(obj.reports)
                rp = obj.reports(k);
                M = [M; rp.results];
            end
            obj.results = M;
        end
        
        function stack_reports(obj)
            obj.stack_results();
            obj.stack_metrics();
            % obj.stack_statistics();
        end

        function aggregate_reports(obj)
            fprintf("MLTRAININGREPORT:[Aggregating reports...]\n");
            
            t1 = datetime;

            tbl_metrics = obj.metrics;

            metric_varnames = tbl_metrics.Properties.VariableNames;
            metric_weights = tbl_metrics.num_test;
            tmp = sum(tbl_metrics{:,1:3}, 1);

            mstacked = tbl_metrics{:,4:end};
            avgs = metric_weights(:)'*mstacked/sum(metric_weights);
            % avgs = round(avgs, 6);

            m_aggs = [tmp, table2array(compute_ml_metrics(obj.results))];
            m_weighted = [tmp, avgs];
            
            tbl_combined = array2table([m_aggs; m_weighted],"VariableNames", metric_varnames);
            tbl_combined.Properties.RowNames = ["aggregated", "weighted"];
            obj.metrics_combined = tbl_combined;

            t2 = datetime;
            fprintf("\t ...Done! (%.2f seconds)\n", seconds(t2 - t1));
        end
    end
end


%% SUPPORT FUNCTIONS

function out = compute_ml_metrics(data)

    if istable(data)
        data = table2array(data);
    end

    num_cols = size(data,2);
    predictions = data(:,1:num_cols/2);
    targets = data(:,(num_cols/2 + 1):end);

    funcs.sbp = @(A) max(A, [], 2);
    funcs.dbp = @(A) min(A, [], 2);

    % test_size = size(predictions, 1);
    % tbl_main = table(test_size);

    tbl_fiducials = [];

    for fn = ["sbp", "dbp"]
        X = funcs.(fn)(predictions);
        Y = funcs.(fn)(targets);
        m2 = compute_correlation(X, Y);
        m3 = compute_errors_fiducials(X, Y);
        tbl = [m2, m3];
        tbl.Properties.VariableNames = fn + "_" + tbl.Properties.VariableNames;

        tbl_fiducials = [tbl_fiducials, tbl];
    end

    tbl_waveform = compute_metrics_waveform(predictions, targets);

    out = [tbl_waveform, tbl_fiducials];
end

%% other helper functions

function out = compute_metrics_waveform(predictions, targets)
    DP = dot(predictions, targets, 2);
    normA = sqrt(dot(predictions, predictions, 2));
    normB = sqrt(dot(targets, targets, 2));

    cs = DP./(normA.*normB);
    % csd = 1 - cs;
    acs = sum(cs)/size(cs, 1);

    err = predictions - targets;

    mae = sum(abs(err),2)/size(err,2);
    rmse = sqrt(sum(err.^2,2)/size(err,2));

    amae = sum(mae)/size(err,1);
    armse = sum(rmse)/size(err,1);

    keys = ["acs", "amae", "armse"];
    values = [acs, amae, armse];
    % values = round(values, 6);

    out = format_output(keys, values, "table");

end

function out = compute_correlation(predictions, targets)
    N = numel(predictions);
    N_default = 1;
    num_trials = min(50000, factorial(N));

    conc_func = @(X,Y, pcc) 2*pcc*std(X)*std(Y)/(std(X)^2 + std(Y)^2 + (mean(X) - mean(Y))^2);
    r_func = @(X,Y) (dot(X,Y) - N*mean(X)*mean(Y))/((N-1)*std(X)*std(Y));
    t_func = @(r) r*sqrt(N - 2)/sqrt(1 - r^2);
    p_func = @(t) 2*(1 - tcdf(abs(t), N-2));

    [~, pv_default] = corr(predictions, targets, 'Type', 'Pearson');

    r = r_func(predictions,targets);
    t = t_func(r);
    if N >= N_default
        % large sample -> use asymptotic method (central limit theorem)
        pv = p_func(t);
        fprintf("N=%d, pv=%.6f (pv_default=%.6f)\n", N, pv, pv_default);
    else
        % small sample -> use permutation test        
        counts = 0;
        for k = 1:num_trials
            idx = randperm(N);
            Yk= targets(idx);
            rk = r_func(predictions,Yk);
            tk = t_func(rk);
            counts = counts + (abs(tk)>=abs(t));
        end
        pv = counts/num_trials;
        fprintf("N=%d, pv=%.6f (pv_default=%.6f), counts=%d\n", N, pv, pv_default, counts);
    end

    r2 = r^2;
    cc = conc_func(predictions,targets,r); % concordance coefficient (Lawrence I-Kuei Lin)

    keys = ["r2", "pv", "cc"];
    values = [r2, pv, cc];
    % values = round(values, 6);

    out = format_output(keys, values, "table");
end

function out = compute_errors_fiducials(predictions, targets)

    err = predictions - targets;
    
    mean_err = mean(err);
    ci_low = prctile(err, 2.5);
    ci_high = prctile(err, 97.5);

    loa_keys = ["me", "CiLow", "CiHigh"];
    loa = [mean_err, ci_low, ci_high];

    ae = abs(err);

    bounds = [5, 10, 15];
    tol_keys = ["tol05", "tol10", "tol15"];
    func = @(e, b) sum(e < b)/numel(e) * 100;
    percents = arrayfun(@(b) func(ae, b) , bounds);

    mae = mean(ae);
    sdae = std(ae);
    wd = compute_wd_1d(predictions, targets);

    keys = [loa_keys, "mae", "sd", tol_keys, "wd"];
    values = [loa, mae, sdae, percents, wd];
    % values = round(values, 6);

    out = format_output(keys, values, "table");
end

function out = format_output(keys, values, type)
    arguments
        keys
        values
        type (1,1) string
    end

    type = lower(type);

    if strcmpi(type,"values")
        out = values;
    elseif strcmpi(type,"cell")
        out = {keys(:), values(:)};
    elseif ismember(type, ["dictionary", "dict"])
        out = dictionary(keys, values);
    elseif ismember(type, ["table", "tbl", "tabular"])
        out = array2table(values, "VariableNames", keys);
    elseif strcmpi(type, "struct")
        for idx = 1:numel(keys)
            key = keys(idx);
            val = values(idx);
            out.(key) = val;
        end
    end

end

function wd = compute_wd_1d(X, Y)
    % Wasserstein-1 distance between two univariate distributions
    % See https://gchron.copernicus.org/articles/5/263/2023/#bib1.bibx11
    % for interpretation
    low = floor(min([X; Y])) - 0.5;
    high = ceil(max([X; Y])) + 0.5;
    dx = 0.1;

    edges = low:dx:high;
    p1 = histcounts(X, edges)/numel(X);
    p2 = histcounts(Y, edges)/numel(Y);

    c1 = cumsum(p1);
    c2 = cumsum(p2);

    wd = sum(abs(c1 - c2)*dx);
end