function T = spm_vif(spm_mat_path, varargin)
% SPM_VIF  Compute VIFs for regressors in an SPM design matrix.
%
% Usage:
%   T = spm_vif('SPM.mat');
%   T = spm_vif('SPM.mat', 'excludePatterns', {'constant','DCT','rp_'});
%
% Inputs:
%   spm_mat_path      - path to SPM.mat
%
% Name-Value options:
%   'excludePatterns' - cellstr of case-insensitive substrings; any regressor
%                       whose name contains one of these will be EXCLUDED
%                       from the VIF report (default: {}).
%   'display'         - true/false to print a summary and open a figure (default: true)
%
% Output:
%   T  - table with variables: Name, Index, R2, Tolerance, VIF, Flag
%
% Notes:
%   - VIF_j = 1 / (1 - R^2_j), where R^2_j is from regressing regressor j onto
%     all the other columns of the (filtered) design matrix.
%   - Backslash operator handles rank deficiency (least-squares).
%   - SPM.xX.X is after filtering/orthogonalization: this is what gets estimated.

opts.excludePatterns = {};
opts.display = true;
if ~isempty(varargin)
    for k = 1:2:numel(varargin)
        opts.(varargin{k}) = varargin{k+1};
    end
end

% Load SPM and get design
S = load(spm_mat_path, 'SPM');
SPM = S.SPM;
X   = SPM.xX.X;                % [timepoints x regressors]
names = SPM.xX.name(:);        % cellstr

p = size(X,2);

% Build include mask based on excludePatterns
incl = true(p,1);
if ~isempty(opts.excludePatterns)
    for pat = string(opts.excludePatterns)
        % Use regex match instead of contains
        mask = ~cellfun(@(n) ~isempty(regexp(n, pat, 'once')), names);
        incl = incl & mask;
    end
end

idx_keep = find(incl);
X_keep   = X(:, idx_keep);
names_keep = names(idx_keep);

% Compute VIFs
n_keep = numel(idx_keep);
VIF  = nan(n_keep,1);
R2   = nan(n_keep,1);
tol  = nan(n_keep,1);

for ii = 1:n_keep
    j = ii; % index within keep-set
    y = X_keep(:, j);

    % Skip zero-variance columns
    if var(y) < eps
        VIF(ii) = Inf; R2(ii) = 1; tol(ii) = 0;
        continue;
    end

    others = setdiff(1:n_keep, j);
    Xo = X_keep(:, others);

    % Least-squares fit of y ~ Xo
    beta = Xo \ y;          % handles rank deficiency
    yhat = Xo * beta;

    TSS = sum( (y - mean(y)).^2 );
    RSS = sum( (y - yhat).^2 );
    R2(ii)  = max(0, min(1, 1 - RSS / TSS));
    tol(ii) = max(eps, 1 - R2(ii));
    VIF(ii) = 1 ./ tol(ii);
end

% Assemble table
T = table( names_keep, idx_keep, R2, tol, VIF, ...
    'VariableNames', {'Name','Index','R2','Tolerance','VIF'});

% Add a simple flag column
Flag = strings(n_keep,1);
Flag(VIF > 10) = "VIF>10";
Flag(VIF > 5 & VIF <= 10) = "VIF>5";
Flag(Flag == "") = "OK";
T.Flag = Flag;

% Sort by VIF descending for quick inspection
T = sortrows(T, 'VIF', 'descend');

if opts.display
    % Print a brief summary
    fprintf('\n===== VIF summary (excluded patterns: %s) =====\n', ...
        strjoin(string(opts.excludePatterns), ', '));
    fprintf('Regressors inspected: %d\n', n_keep);
    fprintf('Max VIF: %.2f  | Median VIF: %.2f\n', max(T.VIF), median(T.VIF));
    n5  = sum(T.VIF > 5);
    n10 = sum(T.VIF > 10);
    fprintf('Counts: VIF>5: %d, VIF>10: %d\n', n5, n10);

    % Simple bar plot
    figure('Name','VIF by regressor');
    bar(T.VIF);
    grid on;
    xlabel('Regressor (sorted)'); ylabel('VIF');
    title('VIFs for retained regressors (higher = more multicollinearity)');
    ylim([0, max(ceil(max(T.VIF)), 10)]);
end
end