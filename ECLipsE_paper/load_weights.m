function weights = load_weights(dataDir_spec, layer_size, neurons, rand_num)
    data_name = [dataDir_spec '\lyr' num2str(layer_size) 'n' num2str(neurons) 'test' num2str(rand_num) '.mat']
    load(data_name);
end

