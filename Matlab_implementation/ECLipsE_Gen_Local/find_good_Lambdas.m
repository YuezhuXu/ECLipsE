function [Lambdai, ci, status, Xiprev, Mi] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, algo)
        
        exit = 0;
        diprev = size(Wi, 2);
        di = size(Wi, 1);
        optimization_margin = 1e-8;
        

        Dalphai= diag(alphai);
        Dbetai = diag(betai);
        

        active_idx = find(abs(betai - alphai) >= 1e-20);
        % disp("Number of active indices")
        % length(active_idx)
        fix_idx =  find(abs(diag(Dbetai) - diag(Dalphai)) < 1e-20 );
        di_active = numel(active_idx);
        
        
        if isempty(active_idx)
            % All neurons have beta_i == alpha_i: no nonlinear directions
            % In this case, just return zeros or the appropriate identity/linear structure
            Lambdai = 0*eye(size(Wi,1)); 
            ci = 0;
            status = 'Skip';
            Xiprev = Miprev;
            Mi = Miprev;
            return;
        end


        
        if strcmp(algo, 'Acc')


            % Restrict matrices/vectors to active indices only
            Wi_active       = Wi(active_idx, :);
            Winext_active = Winext(:, active_idx);
            Dalphai_active  = Dalphai(active_idx, active_idx);
            Dbetai_active   = Dbetai(active_idx, active_idx);
            
             
            cvx_begin quiet
                variable ci
                variable Li_gen_active(di_active, 1)
                Lambdai_active = diag(Li_gen_active);
               

                % Construct Schur_X using only active neurons
                 K = [Lambdai_active, ...
                      0.5 * Lambdai_active * (Dalphai_active+Dbetai_active) * Wi_active;
                      0.5 * Wi_active' *  (Dalphai_active+Dbetai_active) * Lambdai_active, ...
                      Miprev + Wi_active' * Dalphai_active * Lambdai_active * Dbetai_active * Wi_active];
                 Target = [(Winext_active') * Winext_active, zeros(di_active, diprev);
                           zeros(diprev, di_active), zeros(diprev)];
                 margin = optimization_margin * trace(K) / size(K,1) * eye(size(K,1));
                 LMI = K - ci * Target - margin;
                 LMI = 0.5 * (LMI + LMI');
            
                minimize(-ci)
                subject to
                    LMI == semidefinite(size(LMI,1)) 
                    ci >= 0
                    Li_gen_active >= 0
            cvx_end


            Li_gen = zeros(size(Wi,1),1);
            Li_gen(active_idx) = Li_gen_active;
            mean(Li_gen_active);
            Li_gen(fix_idx) = 1e2*mean(Li_gen_active); 
            Lambdai = diag(Li_gen); 
            
            


            
            K_test = [diag(Li_gen(active_idx)), ...
                      0.5 * diag(Li_gen(active_idx)) * (Dalphai_active+Dbetai_active) * Wi_active;
                      0.5 * Wi_active' *  (Dalphai_active+Dbetai_active) * diag(Li_gen(active_idx)), ...
                      Miprev + Wi_active' * Dalphai_active * diag(Li_gen(active_idx)) * Dbetai_active * Wi_active];
            Target = [(Winext_active') * Winext_active, zeros(di_active, diprev);
                      zeros(diprev, di_active), zeros(diprev)];
            Schur_X_test = K_test - ci * Target;
            Schur_X_test = 0.5 * (Schur_X_test + Schur_X_test');
            Schur_eig_min = min(eig(Schur_X_test));
            
            
            Xiprev = Miprev + Wi' * Dalphai * Lambdai * Dbetai * Wi;
            Xiprev = (Xiprev+Xiprev')/2;
            Mi = Lambdai - 0.25 * Lambdai * (diag(alphai + betai)) * Wi * pinv(Xiprev) * Wi' * (diag(alphai + betai)) * Lambdai;
            Mi = (Mi+Mi')/2;
            Xiprev_eig_min = min(eig(Xiprev));
            Mi_eig_min = min(eig(Mi));

            schur_ok = all(isfinite(Schur_X_test(:))) && (Schur_eig_min > 0);
            if contains(cvx_status, 'Solved') && schur_ok && ...
                    all(isfinite(Li_gen)) && isfinite(ci) && (ci>=1e-12) && all(Li_gen>=0) && ...
                    all(isfinite(Xiprev(:))) && (Xiprev_eig_min > 0) && ...
                    all(isfinite(Mi(:))) && (Mi_eig_min > 0)
                status = 'Solved';
            else
                status = 'Failed';
            end
            % For numerical stability 
            cvx_clear
        
        elseif strcmp(algo, 'Fast')

            % Restrict matrices/vectors to active indices only
            Wi_active       = Wi(active_idx, :);
            Winext_active = Winext(:, active_idx);
            Dalphai_active  = Dalphai(active_idx, active_idx);
            Dbetai_active   = Dbetai(active_idx, active_idx);

            cvx_begin quiet
                variable ci
                variable li_gen_active nonnegative
                Lambdai_active = li_gen_active * eye(length(active_idx));
            
                % Construct Schur_X using only active neurons
                 K = [Lambdai_active, ...
                      0.5 * Lambdai_active * (Dalphai_active+Dbetai_active) * Wi_active;
                      0.5 * Wi_active' *  (Dalphai_active+Dbetai_active) * Lambdai_active, ...
                      Miprev + Wi_active' * Dalphai_active * Lambdai_active * Dbetai_active * Wi_active];
                 Target = [(Winext_active') * Winext_active, zeros(di_active, diprev);
                           zeros(diprev, di_active), zeros(diprev)];
                 margin = optimization_margin * trace(K) / size(K,1) * eye(size(K,1));
                 LMI = K - ci * Target - margin;
                 LMI = 0.5 * (LMI + LMI');
            
                minimize(-ci)
                subject to
                    LMI == semidefinite(size(LMI,1)) 
                    ci >= 0
                    li_gen_active >= 0
            cvx_end

            Lambdai = li_gen_active * eye(di);
            K_test = [li_gen_active * eye(length(active_idx)), ...
                      0.5 * li_gen_active * eye(length(active_idx)) * (Dalphai_active+Dbetai_active) * Wi_active;
                      0.5 * Wi_active' *  (Dalphai_active+Dbetai_active) * li_gen_active * eye(length(active_idx)), ...
                      Miprev + Wi_active' * Dalphai_active * li_gen_active * eye(length(active_idx)) * Dbetai_active * Wi_active];
            Target = [(Winext_active') * Winext_active, zeros(di_active, diprev);
                      zeros(diprev, di_active), zeros(diprev)];
            Schur_X_test = K_test - ci * Target;
            Schur_X_test = 0.5 * (Schur_X_test + Schur_X_test');
            Schur_eig_min = min(eig(Schur_X_test));

            Xiprev = Miprev + Wi' * Dalphai * Lambdai * Dbetai * Wi;
            Xiprev = (Xiprev+Xiprev')/2;
            Mi = Lambdai - 0.25 * Lambdai * (diag(alphai + betai)) * Wi * pinv(Xiprev) * Wi' * (diag(alphai + betai)) * Lambdai;
            Mi = (Mi+Mi')/2;
            Xiprev_eig_min = min(eig(Xiprev));
            Mi_eig_min = min(eig(Mi));
            
            schur_ok = all(isfinite(Schur_X_test(:))) && (Schur_eig_min > 0);
            if contains(cvx_status, 'Solved') && schur_ok && ...
                    isfinite(li_gen_active) && isfinite(ci) && (ci>=1e-12) && (li_gen_active>=0) && ...
                    all(isfinite(Xiprev(:))) && (Xiprev_eig_min > 0) && ...
                    all(isfinite(Mi(:))) && (Mi_eig_min > 0)
                status = 'Solved';
            else
                status = 'Failed';
            end
            cvx_clear

           
           
           
        
        
        elseif  strcmp(algo, 'CF')
            Lambdai = 2/max(eig((Dalphai+Dbetai) * Wi / (Miprev) * Wi' * (Dalphai+Dbetai))) * eye(di);

            
            
            Xiprev = Miprev + Wi' * Dalphai * Lambdai * Dbetai * Wi;
            Xiprev = (Xiprev+Xiprev')/2;
            Mi  = Lambdai - 0.25 * Lambdai * (Dalphai+Dbetai) * Wi * pinv(Xiprev) * Wi' * (Dalphai+Dbetai) * Lambdai;
            Mi = (Mi+Mi')/2;

            fn = Winext * pinv(Mi) * Winext';
            ci = 1/max(eig(fn));

            Schur_X_test = [Lambdai - ci * (Winext') * Winext, ...
                           0.5 * Lambdai * (Dalphai+Dbetai) * Wi;
                           0.5 * Wi' *  (Dalphai+Dbetai) * Lambdai, ...
                           Miprev + Wi' * Dalphai * Lambdai * Dbetai * Wi];
            Schur_X_test = 0.5 * (Schur_X_test + Schur_X_test');
            Schur_eig_min = min(eig(Schur_X_test));
            schur_tol = 1e-14 + 1e-10 * max(1, norm(Schur_X_test, 2));
            Xiprev_eig_min = min(eig(Xiprev));
            Mi_eig_min = min(eig(Mi));
            if all(isfinite(Lambdai(:))) && isfinite(ci) && (ci >= 0) && ...
                    all(isfinite(Schur_X_test(:))) && (Schur_eig_min >= -schur_tol) && ...
                    all(isfinite(Xiprev(:))) && (Xiprev_eig_min > 0) && ...
                    all(isfinite(Mi(:))) && (Mi_eig_min > 0)
                status = 'Solved';
            else
                status = 'Failed';
            end

        else
            error('The algorithm chosen is invalid.')
        
        end


    
       
 


end
