close all
clear

csv_dir = "D:\PviProject\artifacts\_final_ss\s09-cnn-img-to-waveform\history";
all_files = dir(fullfile(csv_dir, "*.csv"));

all_files = all_files(:)';

for file = all_files
    filepath = fullfile(file.folder, file.name);
    
    % Read file as string
    content = fileread(filepath);
    
    % Replace tensor( with nothing
    content = replace(content, 'tensor(', '');
    
    % Replace ), with nothing  
    content = replace(content, ')', '');
    
    % Save back to same file
    fid = fopen(filepath, 'w');
    fprintf(fid, '%s', content);
    fclose(fid);
end