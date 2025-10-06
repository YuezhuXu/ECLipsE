%%
clear
clc



% go one level up
ROOT = '..\';
addpath(genpath(fullfile(ROOT, 'ECLipsE_Gen_Local_matlab/utils/')))

dataDir = fullfile(ROOT, 'datasets_ECLipsE_Gen_Local');

%% Random NN
% Set 1: Small set
lyrs = [5,10,15,20,25];
neurons = [10,20,40,60];
% Set 2: Large set
% lyrs = [30, 40, 50, 60, 70];
% neurons = [60, 80, 100, 120];


center = [0.4; 1.8; -0.5; -1.3; 0.9];
epsilon = 1;


% algo choose from Acc, Fast, CF(only for alphai.*betai>=0)
algo = "Acc";
actv = 'relu'; % leakyrelu para = 0.01 by default


data_ini = zeros(length(neurons), length(lyrs));
Lip_est = array2table(data_ini, 'RowNames', cellstr(string(neurons)), 'VariableNames', cellstr(string(lyrs)));
Time_used = array2table(data_ini, 'RowNames', cellstr(string(neurons)), 'VariableNames', cellstr(string(lyrs)));





cvx_solver_settings( ...
        'gaptol',1e-12,'inftol',1e-12,'steptol',1e-12, ...
        'predcorr',1,'vers',2,'scale_data',0,'printyes',1);
cvx_precision best
% cvx_solver sdpt3
for lyr = lyrs
    for n = neurons
        lyr
        n
        clearvars -except lyr n center epsilon algo actv ...
            dataDir lyrs neurons Lip_est Time_used

        % for random set
        datadir_spec = [dataDir '\random'];
        data = load_weights(datadir_spec, lyr, n);

        weights = data.weights;
        biases = data.biases;
        biases = cellfun(@transpose, biases, 'UniformOutput', false);
        [Lip, time_used, ext] = Get_Lip_estimates(weights, biases, actv, center, epsilon, algo)

        if ext == 0
            Lip_est{num2str(n), num2str(lyr)} = Lip;
            Time_used{num2str(n), num2str(lyr)} = time_used;
        end

   end
end

%% Varying radius for Lipschitz tightness observation
lyrs = [5, 30, 60]; 
n = 128;


center = [0.4; 1.8; -0.5; -1.3; 0.9];
radius = [5, 1, 1/5, 1/5^2, 1/5^3, 1/5^4, 1/5^5];

% algo choose from Acc, Fast, CF(only for alphai.*betai>=0)
algos = {"Acc" "Fast" "CF"};
actv = 'leakyrelu'; 



vn = matlab.lang.makeValidName("r" + string(radius));

data_ini = zeros(length(algos), length(radius));
Lip_est = array2table(data_ini, 'RowNames', cellstr(string(algos)), 'VariableNames', cellstr(string(vn)));
Time_used = array2table(data_ini, 'RowNames', cellstr(string(algos)), 'VariableNames', cellstr(string(vn)));


cvx_solver_settings( ...
        'gaptol',1e-12,'inftol',1e-12,'steptol',1e-12, ...
        'predcorr',1,'vers',2,'scale_data',0,'printyes',1);
cvx_precision best
% cvx_solver sdpt3
for lyr = lyrs
    lyr
    clearvars -except lyr n center epsilon algo actv ...
        dataDir lyrs neurons Lip_est Time_used radius algos ROOT
    for a = 1:length(algos)
        algo = algos{a};
        ind = 0;
        for epsilon = radius 
            algo
            epsilon
            ind = ind + 1;
            datadir_spec = [dataDir '\random'];
            data = load_weights(datadir_spec, lyr, n);

            weights = data.weights;
            biases = data.biases;
            biases = cellfun(@transpose, biases, 'UniformOutput', false);
            [Lip, time_used, ext] = Get_Lip_estimates(weights, biases, actv, center, epsilon, algo)

            if ext == 0
                col = matlab.lang.makeValidName("r" + string(epsilon));
                Lip_est{algo, col} = Lip;
                Time_used{algo,col} = time_used;
            else
                break
            end
        end
    end
    
end



%% MNIST
models_str =  {'base', 'jr'};
radius = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256];
ss = 20;



algo = "Fast";
% MNIST elu
actv = 'elu'; 


cvx_solver_settings( ...
        'gaptol',1e-12,'inftol',1e-12,'steptol',1e-12, ...
        'predcorr',1,'vers',2,'scale_data',0,'printyes',1);
cvx_precision best
% cvx_solver sdpt3
for num = 1:length(models_str)
    model_str = models_str{num}; 
    data_file = [dataDir '\MNIST\trained_NN\mnist_' model_str '.mat'];


    data = load(data_file);
    weights = data.weights;
    biases = data.biases;
    biases = cellfun(@transpose, biases, 'UniformOutput', false);

    vn = matlab.lang.makeValidName("r" + string(radius));
    result_ini = zeros(ss,length(radius));
    Lip_est = array2table(result_ini, 'VariableNames', cellstr(vn));

    for sample = 1:ss
        sample
        % Fix sampling seed such that base and jr estimate Lip on the same points
        rng(sample*77+9); 
        center = rand(784,1);

        for epsilon = radius
            epsilon
            [Lip, time_used, ext] = Get_Lip_estimates(weights, biases, actv, center, epsilon, algo)

            if ext == 0
                col = matlab.lang.makeValidName("r" + string(epsilon));
                Lip_est{sample, col} = Lip
            end

        end 

    end
end













