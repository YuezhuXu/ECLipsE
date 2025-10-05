function [Lambdai, ci, status, Xiprev, Mi] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, algo)
        
        exit = 0;
        diprev = size(Wi, 2);
        di = size(Wi, 1);
        

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
                 Schur_X = [Lambdai_active - ci * (Winext_active') *Winext_active, ...
                           0.5 * Lambdai_active * (Dalphai_active+Dbetai_active) * Wi_active;
                           0.5 * Wi_active' *  (Dalphai_active+Dbetai_active) * Lambdai_active, ...
                           Miprev + Wi_active' * Dalphai_active * Lambdai_active * Dbetai_active * Wi_active];
            
                minimize(-ci)
                subject to
                    Schur_X - 1e-15 * eye(size(Schur_X,1)) == semidefinite(size(Schur_X,1)) 
                    ci >= 0
                    Li_gen_active >= 0
            cvx_end
            cvx_clear


            Li_gen = zeros(size(Wi,1),1);
            Li_gen(active_idx) = Li_gen_active;
            mean(Li_gen_active);
            Li_gen(fix_idx) = min(1e20, 1e2*mean(Li_gen_active)); 
            Lambdai = diag(Li_gen); 
            
            


            
            ci
            Schur_eig_min = min(eig(Schur_X))
            
            
            Xiprev = Miprev + Wi' * Dalphai * Lambdai * Dbetai * Wi;
            Xiprev = (Xiprev+Xiprev')/2 + 1e-30*eye(size(Wi,2));
            Mi = Lambdai - 0.25 * Lambdai * (diag(alphai + betai)) * Wi * pinv(Xiprev) * Wi' * (diag(alphai + betai)) * Lambdai;
            Mi = (Mi+Mi')/2+ 1e-30*eye(size(Lambdai));
            Mi_eig_min = min(eig(Mi));


            if (Schur_eig_min>-1e-6) && (ci>=1e-12) && all(Li_gen>=0) && (Mi_eig_min>-1e-6)
                status = 'Solved';
            else
                status = 'Failed';
            end


            % For numerical stability 
            Lambdai = value(Lambdai);
            Lambdai = min(1e20,value(Lambdai));
        
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
                 Schur_X = [Lambdai_active - ci * (Winext_active') * Winext_active, ...
                           0.5 * Lambdai_active * (Dalphai_active+Dbetai_active) * Wi_active;
                           0.5 * Wi_active' *  (Dalphai_active+Dbetai_active) * Lambdai_active, ...
                           Miprev + Wi_active' * Dalphai_active * Lambdai_active * Dbetai_active * Wi_active];
            
                minimize(-ci)
                subject to
                    Schur_X - 1e-15 * eye(size(Schur_X,1)) == semidefinite(size(Schur_X,1)) 
                    ci >= 0
                    li_gen_active >= 0
            cvx_end


            li_gen_active = min(1e20, li_gen_active);
            
            Lambdai = li_gen_active * eye(di);
            cvx_clear
            ci
            Schur_eig_min = min(eig(Schur_X))

            Xiprev = Miprev + Wi' * Dalphai * Lambdai * Dbetai * Wi;
            Xiprev = (Xiprev+Xiprev')/2 + 1e-30*eye(size(Wi,2));
            Mi = Lambdai - 0.25 * Lambdai * (diag(alphai + betai)) * Wi * pinv(Xiprev) * Wi' * (diag(alphai + betai)) * Lambdai;
            Mi = (Mi+Mi')/2+ 1e-30*eye(size(Lambdai));
            Mi_eig_min = min(eig(Mi));
            

            

            if (Schur_eig_min>-1e-6) && (ci>=1e-12) && (li_gen_active>=0) && (Mi_eig_min>-1e-6)
                status = 'Solved';
            else
                status = 'Failed';
            end

           
           
           
        
        
        elseif  strcmp(algo, 'CF')
            Lambdai = 2/max(eig((Dalphai+Dbetai) * Wi / (Miprev) * Wi' * (Dalphai+Dbetai))) * eye(di);

            
            
            Xiprev = Miprev + Wi' * Dalphai * Lambdai * Dbetai * Wi;
            Xiprev = (Xiprev+Xiprev')/2 + 1e-30*eye(size(Wi,2));
            Mi  = Lambdai - 0.25 * Lambdai * (Dalphai+Dbetai) * Wi * pinv(Xiprev) * Wi' * (Dalphai+Dbetai) * Lambdai;
            Mi = (Mi+Mi')/2+ 1e-30*eye(size(Lambdai));

            fn = Winext * pinv(Mi) * Winext';
            ci = 1/max(eig(fn));

            status = 'Solved';

            ci

        else
            error('The algorithm chosen is invalid.')
        
        end


    
       
 


end