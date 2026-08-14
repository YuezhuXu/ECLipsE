function [global_low, global_high] = ActvSlopeRange_one_global(actType)

    alpha = 0.01; % LeakyReLU param
    elu_a = 1.0;  % ELU param
    
    switch lower(actType)
        case 'relu'
            global_low = 0; global_high = 1;
        case 'leakyrelu'
            global_low = alpha; global_high = 1;
        case 'sigmoid'
            global_low = 0; global_high = 0.25;
        case 'tanh'
            global_low = 0; global_high = 1;
        case 'elu'
            global_low = 0; global_high = elu_a;
        case 'silu'
            global_low = -0.0734; global_high = 1.1;
        case 'swish'
            global_low = 0; global_high = 1.1;
        case 'softplus'
            global_low = 0; global_high = 1;
        otherwise
            error('Unknown activation "%s".', actType);
    end
end
