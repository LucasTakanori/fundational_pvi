clear
close all

ds_root = "D:\PviProject\datasets";

ds_subdir = fullfile(ds_root,"longitudinal");

interps_dir = fullfile(ds_subdir,"_interps");
interps_subject = fullfile(interps_dir,"subjects");

bp_files = dir(fullfile(interps_subject,"*bp.csv"));

matName = "bp_interps.mat";
matPath = fullfile(ds_root, matName);