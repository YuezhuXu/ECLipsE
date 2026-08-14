function [Lip_est, time_used, trivial_Lip] = ECLipsE(weights)
    l = size(weights, 2);
    
    for i = 1:l
        eval(['W' num2str(i) '= weights{' num2str(i) '};']);
        eval(['d' num2str(i) '=' 'size(W' num2str(i) ',1);']);
    end

    alpha = 0; 
    beta = 1;
    p = alpha * beta;
    m = (alpha + beta) / 2;

    d0 = size(W1, 2);
    l0 = 0;

    tic;
    d_cum = 0;
    Xi_prev = eye(d0);
    optimization_margin = 1e-8;

    for i = 1:l-1
       fprintf('ECLipsE stage %d of %d.\n', i, l-1);
       di = eval(['d' num2str(i)]); % d1
       Wi = eval(['W' num2str(i)]); % W1
       Wi_next = eval(['W' num2str(i + 1)]); % W2
       
       % Because p = 0, M_i = X_i 
       Inv_Xi_prev = inv(Xi_prev);  

       Ki = (m^2 * Wi * Inv_Xi_prev * Wi');
       Ki = (Ki + Ki') / 2;

       cvx_begin quiet
           variable s
           variable Li_gen(di, 1)
           Li = diag(Li_gen);
           K = [Li, Li * sqrtm(Ki);
                sqrtm(Ki) * Li, eye(di)];
           Target = [Wi_next' * Wi_next, zeros(di);
                     zeros(di), zeros(di)];
           margin = optimization_margin * trace(K) / size(K,1) * eye(size(K,1));
           LMI = K - s * Target - margin;
           LMI = 0.5 * (LMI + LMI');
           minimize(-s)
           subject to
               LMI == semidefinite(2*di)
               s >= 1e-20
               Li >= 0
       cvx_end

       Li = diag(Li_gen);
       K_test = [Li, Li * sqrtm(Ki);
                 sqrtm(Ki) * Li, eye(di)];
       Target = [Wi_next' * Wi_next, zeros(di);
                 zeros(di), zeros(di)];
       Schur_X_test = K_test - s * Target;
       Schur_X_test = 0.5 * (Schur_X_test + Schur_X_test');
       Schur_eig_min = min(eig(Schur_X_test));

       if ~contains(cvx_status, 'Solved') || ~isfinite(s) || ~all(isfinite(Li_gen)) || ...
               s <= 1e-20 || any(Li_gen < 0) || ...
               ~all(isfinite(Schur_X_test(:))) || Schur_eig_min <= 0
           disp('No Feasible Solution!');
           break;
       end
        
       Xi = Li - m^2 * Li * Wi * Inv_Xi_prev * Wi' * Li;
       Xi = 0.5 * (Xi + Xi');
       if ~all(isfinite(Xi(:))) || min(eig(Xi)) <= 0
           disp('Numerical Issues!');
           break;
       end
       Xi_prev = Xi;
       d_cum = d_cum + di;
    end 
    
    Wl = eval(['W' num2str(l)]);
    oneoverF = max(eig((Wl' * Wl) / Xi_prev));
    Lip_sq_est = oneoverF;

    time_used = toc;
    
    Lip_est = sqrt(Lip_sq_est);
    trivial_Lip_sq = 1;
    for i = 1:l
        eval(['trivial_Lip_sq = trivial_Lip_sq * norm(W' num2str(i) ')^2;']);
    end

    trivial_Lip = sqrt(trivial_Lip_sq);
    fprintf('The Lipschitz estimation is %.10g. Estimation is completed in %.6g seconds.\n', Lip_est, time_used);
end


