%%
clear
clc



% go one level up
ROOT = '../';

dataDir = fullfile(ROOT, 'datasets');

%% Random NN
% Set 1: Small set
lyrs = [5,10,15,20,25];
neurons = [10,20,40,60];
% % Set 2: Large set
lyrs = [30, 40, 50, 60, 70];
neurons = [60, 80, 100, 120];


center = [0.4; 1.8; -0.5; -1.3; 0.9];
epsilon = 1;


% algo choose from Acc, Fast, CF(only for alphai.*betai>=0)
algo = "Acc";
actv = 'relu'; % leakyrelu para = 0.1


data_ini = zeros(length(neurons), length(lyrs));
Lip_est = array2table(data_ini, 'RowNames', cellstr(string(neurons)), 'VariableNames', cellstr(string(lyrs)));
Time_used = array2table(data_ini, 'RowNames', cellstr(string(neurons)), 'VariableNames', cellstr(string(lyrs)));





cvx_solver_settings('eps',1e-15,'gaptol',1e-15);
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

%% Varying radius for Lipschitz tightness observation (not fixed yet)
lyrs = [5,30,60];
n = 128


center = [0.4; 1.8; -0.5; -1.3; 0.9];
radius = [1/1024, 1/256, 1/64, 1/16, 1/4, 1];


% algo choose from Acc, Fast, CF(only for alphai.*betai>=0)
algo = "Acc"; % "Fast" "CF"
actv = 'leakyrelu'; % leakyrelu para = 0.1. relu, leakyrelu will go to reduced skip case when radius is small.



data_ini = zeros(length(radius), length(lyrs));
Lip_est = array2table(data_ini, 'VariableNames', cellstr(string(lyrs)));
Time_used = array2table(data_ini, 'VariableNames', cellstr(string(lyrs)));




cvx_solver_settings('eps',1e-15,'gaptol',1e-15);
cvx_precision best
% cvx_solver sdpt3
for lyr = lyrs
    lyr
    clearvars -except lyr n center epsilon algo actv ...
        dataDir lyrs neurons Lip_est Time_used radius

    ind = 0;
    for epsilon = radius
        epsilon
        ind = ind + 1;
        datadir_spec = [dataDir '\random'];
        data = load_weights(datadir_spec, lyr, n);

        weights = data.weights;
        biases = data.biases;
        biases = cellfun(@transpose, biases, 'UniformOutput', false);
        [Lip, time_used, ext] = Get_Lip_estimates(weights, biases, actv, center, epsilon, algo)

        if ext == 0
            Lip_est{ind, num2str(lyr)} = Lip;
            Time_used{ind, num2str(lyr)} = time_used;
        end
    end
end


%% MNIST and Fashion MNIST
models_str = {'base', 'jr'};
epsilons = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64];
ss = 20;



algo = "Fast";
actv = 'silu'; 


cvx_solver_settings('eps',1e-15,'gaptol',1e-15);
cvx_precision best
for num = 1:length(models_str)
    model_str = models_str{num};   
    % data_file = [dataDir '\FashionMNIST\trained_NN\fmnist_' model_str '.mat'];
    data_file = [dataDir '\MNIST\trained_NN\mnist_' model_str '.mat'];
    data = load(data_file);
    weights = data.weights;
    biases = data.biases;
    biases = cellfun(@transpose, biases, 'UniformOutput', false);

    rr = 1./epsilons;
    vn = matlab.lang.makeValidName("r" + string(rr));
    result_ini = zeros(ss,length(epsilons));
    Lip_est = array2table(result_ini, 'VariableNames', cellstr(vn));

    for sample = 1:ss
        sample
        % Fix sampling seed such that base and jr estimate Lip on the same points
        rng(sample*123);
        center = rand(784,1);

        for epsilon = epsilons
            epsilon
            [Lip, time_used, ext] = Get_Lip_estimates(weights, biases, actv, center, epsilon, algo)

            if ext == 0
                col = matlab.lang.makeValidName("r" + string(1/epsilon));
                Lip_est{sample, col} = Lip
            end

        end 

    end
    writetable(Lip_est, [ROOT '/results/robust_training/MNIST_Lip_' model_str '.csv'])
end













