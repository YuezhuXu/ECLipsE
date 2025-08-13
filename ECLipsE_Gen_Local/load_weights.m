function params = load_weights(dataDir_spec, layer_size, neurons)
    % Construct filename, e.g. 'weights_lyr3n40.mat'
    fname = sprintf('lyr%dn%d.mat', layer_size, neurons);
    
    fullpath = fullfile(dataDir_spec, fname);

    params = load(fullpath);

end
