% convert_hnei_to_csv.m
% Run this script in MATLAB to convert all HNEI .mat files to .csv format.

clc; clear;

% Make sure you run this script from the root of your project directory:
% c:\Users\ss778\Downloads\battery-health-cnn-tcn-lstm-main\battery-health-cnn-tcn-lstm-main

baseDir = fullfile('data', 'raw', 'HalfCycle');
outDir  = fullfile('data', 'raw', 'HalfCycle', 'csv');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

matFiles = dir(fullfile(baseDir, '**', '*.mat'));

fprintf('Found %d .mat files.\n', length(matFiles));

for i = 1:length(matFiles)
    filePath = fullfile(matFiles(i).folder, matFiles(i).name);
    [~, cellName, ~] = fileparts(matFiles(i).name);

    % Determine SOC condition from folder path
    parts = strsplit(matFiles(i).folder, filesep);
    socLabel = '';
    for p = 1:length(parts)
        if startsWith(parts{p}, 'SOC_')
            socLabel = parts{p};
            break;
        end
    end

    fprintf('\n[%d/%d] Processing %s (SOC: %s)...\n', i, length(matFiles), cellName, socLabel);

    try
        S = load(filePath);
        varNames = fieldnames(S);
        varName = varNames{1};
        dataCell = S.(varName);

        allTables = {};
        nRows = size(dataCell, 1);

        for r = 2:nRows
            opName = dataCell{r, 1};
            tbl = dataCell{r, 3};

            if istable(tbl) && height(tbl) > 0
                tbl.Operation = repmat(string(opName), height(tbl), 1);
                tbl.BlockIndex = repmat(r-1, height(tbl), 1);
                allTables{end+1} = tbl;
            end
        end

        if isempty(allTables)
            fprintf('  WARNING: No table data found. Skipping.\n');
            continue;
        end

        combined = vertcat(allTables{:});
        
        csvName = sprintf('%s_%s.csv', cellName, socLabel);
        csvPath = fullfile(outDir, csvName);

        if isfile(csvPath)
            error('Refusing to overwrite existing CSV: %s', csvPath);
        end
        writetable(combined, csvPath);
        fprintf('  Saved %s (%d rows)\n', csvPath, height(combined));

    catch ME
        fprintf('  ERROR: %s\n', ME.message);
    end
end

fprintf('\n=== Conversion complete. CSVs saved to %s ===\n', outDir);
