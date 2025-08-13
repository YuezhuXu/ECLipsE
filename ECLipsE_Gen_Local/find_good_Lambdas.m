function [Lambdai, ci, status] = find_good_Lambdas(Wi, Winext, Miprev, alphai, betai, algo)
        
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
            Lambdai = 0*eye(size(Wi,1)); % or diag(zeros(size(Wi,1),1))
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
                
                % Winext2 = Winext' * Winext;
                % Winext2active = Winext2(active_idx, active_idx)

                % Construct Schur_X using only active neurons
                 Schur_X = [Lambdai_active - ci * (Winext_active') *Winext_active, ...
                           0.5 * Lambdai_active * (Dalphai_active+Dbetai_active) * Wi_active;
                           0.5 * Wi_active' *  (Dalphai_active+Dbetai_active) * Lambdai_active, ...
                           Miprev + Wi_active' * Dalphai_active * Lambdai_active * Dbetai_active * Wi_active];
            
                minimize(-ci)
                subject to
                    Schur_X - 1e-15 * eye(size(Schur_X,1)) == semidefinite(size(Schur_X,1))
                    ci >= 1e-20
                    Li_gen_active >= 0
            cvx_end
            % cvx_status
            cvx_clear

            % If you want to reconstruct the full vector for later steps:
            Li_gen = zeros(size(Wi,1),1);
            Li_gen(active_idx) = Li_gen_active;
            mean(Li_gen_active);
            Li_gen(fix_idx) = min(1e15, 1e2*mean(Li_gen_active)); % mean(Li_gen_active);
            Lambdai = diag(Li_gen); % (full-size, zeros in inactive positions)
            
            


            
            ci
            Schur_eig_min = min(eig(Schur_X))
            
            
            

            if (Schur_eig_min>0) && (ci>=0) && all(Li_gen>=0)
                status = 'Solved';
            else
                status = 'Failed';
            end


            % For numerical stability 
            Lambdai = value(Lambdai);
            Lambdai = min(1e15,value(Lambdai));
        
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
                
                % Winext2 = Winext' * Winext;
                % Winext2active = Winext2(active_idx ,active_idx);

                % Construct Schur_X using only active neurons
                 Schur_X = [Lambdai_active - ci * (Winext_active') * Winext_active, ...
                           0.5 * Lambdai_active * (Dalphai_active+Dbetai_active) * Wi_active;
                           0.5 * Wi_active' *  (Dalphai_active+Dbetai_active) * Lambdai_active, ...
                           Miprev + Wi_active' * Dalphai_active * Lambdai_active * Dbetai_active * Wi_active];
            
                minimize(-ci)
                subject to
                    Schur_X - 1e-15 * eye(size(Schur_X,1)) == semidefinite(size(Schur_X,1)) 
                    ci >= 1e-20
                    li_gen_active >= 0 
            cvx_end



            Lambdai = li_gen_active * eye(di);
            cvx_clear
            ci
            Schur_eig_min = min(eig(Schur_X));
            

            

            if (Schur_eig_min>0) && (ci>=0) && li_gen_active>=0
                status = 'Solved';
            else
                status = 'Failed';
            end

           
           
           
        
        
        elseif  strcmp(algo, 'CF')
            
            
            

            Lambdai = 2/max(eig((Dalphai+Dbetai) * Wi / (Miprev) * Wi' * (Dalphai+Dbetai))) * eye(di);

            
            
            A22 = Miprev + Wi' * Dalphai * Lambdai * Dbetai * Wi;
            A22_inv = pinv(A22);
            A22_inv = (A22_inv+A22_inv')/2;
            M0  = Lambdai - 0.25 * Lambdai * (Dalphai+Dbetai) * Wi * A22_inv * Wi' * (Dalphai+Dbetai) * Lambdai;
            fn = Winext * pinv(M0) * Winext';
            ci = 1/max(eig(fn));
            % B  = Winext' * Winext + 1e-30 *eye(size(Winext,2));
            % ci = min(eig(M0, B))
            status = 'Solved';

            ci

        else
            error('The algorithm chosen is invalid.')
        
        end


    
       
 


end