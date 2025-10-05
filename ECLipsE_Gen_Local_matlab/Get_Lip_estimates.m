function [Lip, alphas, betas, time_used, ext] = Get_Lip_estimates(weights, biases, actv, centr, epsilon, algo)

        
        N = size(weights, 2);
        
        delta_z_norm2 = epsilon*sqrt(size(centr,1));
        
        d0 = size(weights{1},2);

        Mi = eye(d0); % M0
        value_c = centr;


        tic;

        
        ext = 0;
        skip = 0;

        alphas = {};
        betas = {};

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
            alphas = [alphas alphai];
            betas = [betas betai];

            % go through activation function 
            if i<N-1
                value_c =  eval([actv '(value_c)']);
            end
            



            % CF, check signs.
            if any((alphai .* betai >=0)==0) & strcmp(algo, 'CF')
                error('Alphai and Betai do not have matching signs elementwise so CF algorithmm is not applicable.');
            end 
            % Refine bounds on one side to make abs(alphai+betai) smaller
            if strcmp(algo, 'CF')
                alphai((alphai>=0)) = 0;
                betai((0>=betai)) = 0;
            end



            % Now run ECLipse-Gen-Local-Acc/Fast/CF to obtain Lambdai
            intv = betai-alphai;
            intv_sum = sum(intv);
            [Lambdai, ci, status, Xiprev, Mi] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, algo);            
                

            
            % status
            if strcmp(status, 'Failed')
                disp("Failed in initial try.")
                % Reset skip
                skip = 0;

                if strcmp(algo, 'Fast')
                    alphai((alphai>=0)) = 0;
                    betai((0>=betai)) = 0;
                    [Lambdai, ci, status, Xiprev, Mi] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, 'CF');
                elseif strcmp(algo, 'Acc')
                    disp('Reduce to Fast/CF.')

                    % Fast
                    [Lambdai_fast, ci_fast, status_fast, Xiprev_fast, Mi_fast] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, 'Fast');

                    % CF
                    alphai_CF = alphai;
                    betai_CF = betai;
                    alphai_CF((alphai_CF>=0)) = 0;
                    betai_CF((0>=betai_CF)) = 0;

                    [Lambdai_CF, ci_CF, status_CF, Xiprev_CF, Mi_CF] = find_good_Lambdas(Wi, Winext, Miprev, alphai_CF, betai_CF, 'CF');

                    if strcmp(status_fast, 'Failed') | (ci_CF>=ci_fast)
                        disp('CF is picked.')
                        Lambdai = Lambdai_CF;
                        alphai = alphai_CF;
                        betai = betai_CF;
                        Xiprev = Xiprev_CF;
                        Mi = Mi_CF;
                    else
                        disp('Fast is picked.')
                        Lambdai = Lambdai_fast;
                        Xiprev = Xiprev_Fast;
                        Mi = Mi_Fast;
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



            
            if  min(eig(Xiprev))<0
                ext = -1;
                disp("Xi < 0. Break.")
                break
            end

            disp("Checking Mi")
            min(eig(Mi))


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