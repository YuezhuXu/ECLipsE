clear
clc

dataDir = './datasets';
% Case 1, 2
% lyrs = [2, 5, 10, 20, 30, 50, 75, 100];
% neurons = [20, 40, 60, 80, 100];
% Case 3
% lyrs = [100];
% neurons = [20, 40, 60, 80, 100, 120, 140, 160];
% For mnist
lyrs = [3];
neurons = [100, 200, 300, 400];
% For even wider NN
%lyrs = [50];
%neurons = [150, 200, 300, 400, 500,1000];


%% Experiments 
data_ini = zeros(length(neurons), length(lyrs));
Lip_est = array2table(data_ini, 'RowNames', cellstr(string(neurons)), 'VariableNames', cellstr(string(lyrs)));
Time_used = array2table(data_ini, 'RowNames', cellstr(string(neurons)), 'VariableNames', cellstr(string(lyrs)));
Trivial_results = array2table(data_ini, 'RowNames', cellstr(string(neurons)), 'VariableNames', cellstr(string(lyrs)));

%%
for lyr = lyrs
    for n =  neurons
        for rd  = 1
            n
            clearvars -except lyr n rd lips_rel times_used ...
                dataDir lyrs neurons Lip_est Time_used Trivial_results
            
            % % For random sets (Case 1,2,3 wider NN)
            % datadir_spec = [dataDir '\random'];
            % weights = load_weights(datadir_spec, lyr, n, rd);

            % For mnist 
            datadir_spec = [dataDir '\MNIST'];
            weights = load_weights(datadir_spec, lyr, n, rd);
            
            % Choose the method
            % [lip, time_used, trivial] = ECLipsE(weights)
            [lip, time_used, trivial] = ECLipsE_Fast(weights)

            Lip_est{num2str(n), num2str(lyr)} = lip;
            Time_used{num2str(n), num2str(lyr)} = time_used;
            Trivial_results{num2str(n), num2str(lyr)} = trivial;
        end
   end
end

ratio = Lip_est./Trivial_results;





