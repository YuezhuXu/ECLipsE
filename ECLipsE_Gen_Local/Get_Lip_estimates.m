function [Lip, time_used, ext] = Get_Lip_estimates(weights, biases, actv, center, epsilon, algo)

        tic;

        
        N = size(weights, 2);
        
        delta_z_norm2 = epsilon*sqrt(size(center,1));

        % slope_range_Lip = slope_range_ini;
        d0 = size(weights{1},2);

        Mi = eye(d0); % M0
        value_c = center;


        tic;

        [ori_alpha, ori_beta] = ActvSlopeRange_one_global(actv);
        ext = 0;
        skip = 0;


        for i = 1:N-1
            i

            if skip == 1
                Wiprev = Wi;
                assert(all(abs(alphai - betai) < 1e-20))
            end

            Wi = weights{i};
            % kept for computing the center value
            Wi_ori = Wi; 
            if skip == 1
                Wi = Wi*diag(alphai)*Wiprev;
            end

            Winext = weights{i+1};
            bi = biases{i};

            % Calculate Lipschitz constant for each neuron
            Miprev = Mi;
            Ai = pinv(Miprev) * Wi';
            Li = sqrt(sum(Wi .* Ai', 2));


            value_c = Wi_ori*value_c+bi;
            f_range = [value_c-delta_z_norm2*Li, value_c+delta_z_norm2*Li];
            
            [alphai,betai] = ActvSlopeRange(actv,f_range);

            % go through activation function 
            if i<N-1
                value_c =  eval([actv '(value_c)']);
            end




            % CF, check signs.
            if any((alphai .* betai >=0)==0) & strcmp(algo, 'CF')
                error('Alphai and Betai do not have matching signs elementwise so CF algorithmm is not applicable.');
            end 
            % Check if the refined alphai, betai will help with the estimate
            % If not, undo the refinement
            % Decide on whether to refine slope bounds
            if strcmp(algo, 'CF')
                absum = alphai + betai;
                ori_sum = ori_alpha + ori_beta;
                if any(absum>=ori_sum)
                    disp('Refinement Undone.')
                    alphai = ori_alpha * ones(size(alphai));
                    betai  = ori_beta *ones(size(betai));
                end
            end


            % Now run ECLipse-Gen-Local-Acc/Fast/CF to obtain Lambdai
            [Lambdai, ci, status] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, algo);            
                

            % ci
            % Lambdai
            % Lambdai_max = max(diag(Lambdai))
            % Lambdai_min = min(diag(Lambdai))


            % status
            if strcmp(status, 'Failed')
                disp("Failed in initial try.")
                % Reset skip
                skip = 0;

                if strcmp(algo, 'Fast')
                    absum = alphai + betai;
                    ori_sum = ori_alpha + ori_beta;
                    if any(absum>=ori_sum)
                        disp('Refinement Undone.')
                        alphai = ori_alpha * ones(size(alphai));
                        betai  = ori_beta *ones(size(betai));
                    end
                    [Lambdai, ci, status] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, 'CF');
                elseif strcmp(algo, 'Acc')
                    disp('Reduce to Fast/CF.')
                    % Fast
                    [Lambdai_fast, ci_fast, status_fast] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, 'Fast');

                    % CF
                    alphai_CF = alphai;
                    betai_CF = betai;
                    absum = alphai_CF + betai_CF;
                    ori_sum = ori_alpha + ori_beta;
                    if any(absum>=ori_sum)
                        disp('Refinement Undone.')
                        alphai_CF = ori_alpha * ones(size(alphai));
                        betai_CF  = ori_beta *ones(size(betai));
                    end
                    [Lambdai_CF, ci_CF, status_CF] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, 'CF');

                    if strcmp(status_fast, 'Failed') | (ci_CF>=ci_fast)
                        disp('CF is picked.')
                        Lambdai = Lambdai_CF;
                        alphai = alphai_CF;
                        betai = betai_CF;
                    else
                        disp('Fast is picked.')
                        Lambdai = Lambdai_fast;
                    end
                end
             elseif  strcmp(status, 'Solved')
                % Reset skip
                skip = 0;
                disp('All good.')
            elseif strcmp(status, 'Skip')
                disp("Affine layer. Directly merge with the next layer.")
                skip = 1;
                % directly next iteration.
                continue;
            end


            % slope_itv_final = diag(betai-alphai);
            % disp("Final Slope Interval")
            % slope_itv_final(end-5:end)

            Xiprev = Miprev + Wi' * diag(alphai) * Lambdai * diag(betai) * Wi;
            Xiprev = (Xiprev+Xiprev')/2 + 1e-30*eye(size(Wi,2)); % For numerical stability 

            if  min(eig(Xiprev))<0
                ext = -1;
                disp("Xi < 0. Break.")
                break
            end



            Mi = Lambdai - ...
            0.25 * Lambdai * (diag(alphai + betai)) * Wi * pinv(Xiprev) * Wi' * (diag(alphai + betai)) * Lambdai;
            Mi = (Mi+Mi')/2+ 1e-30*eye(size(Lambdai)); % For numerical stability



        end



        WM = weights{end};
        if skip == 1
            Wiprev = Wi;
            assert(all(abs(alphai - betai) < 1e-20))
            WM = WM * diag(alphai) * Wiprev;
        end

        XMprev = Mi;
        fn = WM * pinv(XMprev) * WM';
        fn = (fn+fn')/2;
        Lip = sqrt(max(eig(fn)));
        time_used = toc;



end 