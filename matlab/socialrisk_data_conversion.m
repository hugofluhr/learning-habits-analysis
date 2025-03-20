% Load the MAT file
data = load('/Users/hugofluhr/phd_local/data/social-risk/analysis/BevBigTable.mat');

% Write out each table to a separate CSV file
if isfield(data, 'BevBigTable')
    writetable(data.BevBigTable, 'BevBigTable.csv');
else
    warning('Variable "BevBigTable" not found.');
end

if isfield(data, 'BevBigTableRaw')
    writetable(data.BevBigTableRaw, 'BevBigTableRaw.csv');
else
    warning('Variable "BevBigTableRaw" not found.');
end

if isfield(data, 'SubjInfo')
    writetable(data.SubjInfo, 'SubjInfo.csv');
else
    warning('Variable "SubjInfo" not found.');
end

if isfield(data, 'WTPTable')
    writetable(data.WTPTable, 'WTPTable.csv');
else
    warning('Variable "WTPTable" not found.');
end