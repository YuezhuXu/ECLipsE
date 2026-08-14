ROOT = fileparts(fileparts(mfilename('fullpath')));

addpath(fullfile(ROOT, 'Matlab implementation', 'ECLipsE'));
addpath(fullfile(ROOT, 'Matlab implementation', 'ECLipsE_Gen_Local'));
addpath(fullfile(ROOT, 'Matlab implementation', 'ECLipsE_Gen_Local', 'utils'));
results = struct();

sample_files = {
    'lyr30n60.mat', ...
    'lyr40n80.mat', ...
    'lyr50n100.mat', ...
    'lyr60n120.mat', ...
    'lyr70n120.mat'};

for case_idx = 1:numel(sample_files)
    case_name = sample_files{case_idx};
    data = load(fullfile(ROOT, 'demo', 'sampleweights', case_name));
    weights = data.weights;

    depth = numel(weights);
    width = max(cellfun(@(W) size(W, 1), weights(1:end-1)));
    fprintf('\nECLipsE case %d of %d: %s. Depth: %d. Width: %d.\n', ...
        case_idx, numel(sample_files), case_name, depth, width);
    fprintf('Running ECLipsE.\n');
    [L_eclipse, t_eclipse, trivial_eclipse] = ECLipsE(weights);
    fprintf('Running ECLipsE-Fast.\n');
    [L_fast, t_fast, trivial_fast] = ECLipsE_Fast(weights);

    results.ECLipsE(case_idx).name = case_name;
    results.ECLipsE(case_idx).depth = depth;
    results.ECLipsE(case_idx).width = width;
    results.ECLipsE(case_idx).ECLipsE.Lip = L_eclipse;
    results.ECLipsE(case_idx).ECLipsE.time = t_eclipse;
    results.ECLipsE(case_idx).ECLipsE.trivial = trivial_eclipse;
    results.ECLipsE(case_idx).Fast.Lip = L_fast;
    results.ECLipsE(case_idx).Fast.time = t_fast;
    results.ECLipsE(case_idx).Fast.trivial = trivial_fast;
end

center = [0.4; 1.8; -0.5; -1.3; 0.9];
epsilon = 1;
actv = 'elu';

for case_idx = 1:numel(sample_files)
    case_name = sample_files{case_idx};
    data = load(fullfile(ROOT, 'demo', 'sampleweights', case_name));
    weights = data.weights;
    biases = cellfun(@transpose, data.biases, 'UniformOutput', false);

    depth = numel(weights);
    width = max(cellfun(@(W) size(W, 1), weights(1:end-1)));
    fprintf('\nECLipsE-Gen-Local case %d of %d: %s. Depth: %d. Width: %d.\n', ...
        case_idx, numel(sample_files), case_name, depth, width);
    fprintf('Running Acc.\n');
    [L_acc, ~, ~, t_acc, ext_acc] = ...
        ECLipsE_Gen_Local(weights, biases, actv, center, epsilon, 'Acc');
    fprintf('Running Fast.\n');
    [L_fast_local, ~, ~, t_fast_local, ext_fast] = ...
        ECLipsE_Gen_Local(weights, biases, actv, center, epsilon, 'Fast');
    fprintf('Running CF.\n');
    [L_cf, ~, ~, t_cf, ext_cf] = ...
        ECLipsE_Gen_Local(weights, biases, actv, center, epsilon, 'CF');

    results.ECLipsE_Gen_Local(case_idx).name = case_name;
    results.ECLipsE_Gen_Local(case_idx).depth = depth;
    results.ECLipsE_Gen_Local(case_idx).width = width;
    results.ECLipsE_Gen_Local(case_idx).Acc.Lip = L_acc;
    results.ECLipsE_Gen_Local(case_idx).Acc.time = t_acc;
    results.ECLipsE_Gen_Local(case_idx).Acc.ext = ext_acc;
    results.ECLipsE_Gen_Local(case_idx).Fast.Lip = L_fast_local;
    results.ECLipsE_Gen_Local(case_idx).Fast.time = t_fast_local;
    results.ECLipsE_Gen_Local(case_idx).Fast.ext = ext_fast;
    results.ECLipsE_Gen_Local(case_idx).CF.Lip = L_cf;
    results.ECLipsE_Gen_Local(case_idx).CF.time = t_cf;
    results.ECLipsE_Gen_Local(case_idx).CF.ext = ext_cf;
end
