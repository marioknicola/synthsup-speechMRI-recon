% Specify folder containing .mat files
folder = '/Users/marioknicola/MSc Project/kspace_mat_512x512';  

% Get list of all .mat files in that folder
files = dir(fullfile(folder, '*.mat'));

if isempty(files)
    warning('No .mat files found in the specified folder.');
else
    for i = 1:length(files)
        fname = fullfile(folder, files(i).name);
        data = load(fname);
        save(fname, '-struct', 'data', '-v7');
        fprintf('Saved %s as v7 format.\n', files(i).name);
    end
end
