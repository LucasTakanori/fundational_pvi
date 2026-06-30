close all
clear

rootdir = "D:\PviProject";
dir_artifacts = fullfile(rootdir,"artifacts","_final_ss");

subdirs = dir(dir_artifacts);
subdirs = subdirs(arrayfun(@(d) (d.isdir && ~ismember(d.name,[".", ".."])), subdirs));
subdirs = subdirs(arrayfun(@(d) (contains(d.name,"waveform")), subdirs));

ds_dir = fullfile(rootdir, "datasets/");

subjects = arrayfun(@(k) "subject" + pad(string(k), 3, "left", "0"),(1:100)');
session = "valsalva";

h5Names = arrayfun(@(subID ) subID + sprintf("_%s_masked.h5",session), subjects);
h5Paths = arrayfun(@(fileName) fullfile(ds_dir, fileName), h5Names);

ds_available = arrayfun(@(path) logical(exist(path, 'file')), h5Paths);
subjects = subjects(ds_available);

h5Paths = h5Paths(ds_available);

for model_idx = 1:numel(subdirs)
    model_name = subdirs(model_idx).name;

    pDir = fullfile(dir_artifacts, model_name, "configs", "_partition");
    pNames = arrayfun(@(subID) subID + "_partition_mappings.csv", subjects);
    pPaths = arrayfun(@(fileName) fullfile(pDir, fileName), pNames);

    rDir = fullfile(dir_artifacts, model_name, "results");
    rNames = arrayfun(@(subID) subID + "_results.csv", subjects);
    rPaths = arrayfun(@(fileName) fullfile(rDir, fileName), rNames);

    fgDir = fullfile(rDir,"_plots");
    mkdir(fgDir)

    fgName = sprintf("results_%s_all",session);
    fPath_global = fullfile(fgDir, fgName)  + ".pdf";
    if exist(fPath_global,'file')
        delete(fPath_global);
    end

    for ds_idx = 1:numel(h5Paths)
        subject = subjects(ds_idx);

        dsName = join([subject, "valsalva"],"-");

        paths.model_name = model_name;
        paths.h5 = h5Paths(ds_idx);
        paths.partition = pPaths(ds_idx);
        paths.results = rPaths(ds_idx);

        disp(paths)

        if contains(subject,"subject063")
            continue
        end

        close all
        fg = plot_bp_sequences(paths);
        sgtitle({model_name, dsName});

        fgPath_local = fullfile(fgDir, dsName);

        exportgraphics(fg, fgPath_local + ".png",...
            "BackgroundColor","white",...
            "ContentType", "image",...
            "Resolution", 300);

        exportgraphics(fg, fPath_global,...
            "BackgroundColor","white",...
            "ContentType", "vector",...
            "Resolution", 150,...
            "Append",true);
    end
end