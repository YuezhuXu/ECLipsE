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

    for i = 1:l-1
       i
       di = eval(['d' num2str(i)]); % d1
       Wi = eval(['W' num2str(i)]); % W1
       Wi_next = eval(['W' num2str(i + 1)]); % W2
       
       % Because p = 0, M_i = X_i 
       M_prev = Xi_prev;
       Inv_M_prev = inv(M_prev);  

       Ki = (m^2 * Wi * Inv_M_prev * Wi');
       Ki = (Ki + Ki') / 2;
       Ki_ep = Ki + (1e-10) * eye(di);

       cvx_begin quiet
           variable s
           variable Li_gen(di, 1)
           Li = diag(Li_gen);
           Schur_X = [Li - s * (Wi_next' * Wi_next), Li * sqrtm(Ki_ep);
                      sqrtm(Ki_ep) * Li, eye(di)];
           minimize(-s)
           subject to
               Schur_X == semidefinite(2*di)
               s >= 1e-20
               Li >= 0
       cvx_end

       if strcmp(cvx_status, 'Infeasible')
           disp('No Feasible Solution!');
           break;
       end
       if s <= 1e-20
           disp('Numerical Issues!');
           break;
       end
        
       Xi = Li - m^2 * Li * Wi * Inv_M_prev * Wi' * Li;
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
end

function [Lip_est, time_used, trivial_Lip] = ECLipsE_Fast(weights)
    % for random weights
    l = size(weights,2);
    % W1 = weights{1};
    for i = 1:l
        eval(['W' num2str(i) '= weights{' num2str(i) '};'])
        %d1 = size(W1,1);
        eval(['d' num2str(i) '=' 'size(W' num2str(i) ',1);'])
    end

    
    alpha = 0; 
    beta = 1;
    p = alpha*beta;
    m = (alpha+beta)/2;
    
    tic;
    d0 = size(W1,2);
    Xi_prev = eye(d0);
    
    
    for i = 1:l-1   
       di = eval('d'+string(i)); %d1
       Wi = eval('W'+string(i)); %W1
    
       % Ai = As.('A'+string(i)); 
       % Bi = Bs.('B'+string(i));
       % Wi = eval('W'+string(i));
       
       % Because p = 0, M_i = X_i 
       M_prev = value(Xi_prev);
       Inv_M_prev = inv(M_prev);  
    
       
       % li_closedform = 1/(2*m^2*max(eig(Wi*Inv_M_prev*Wi')));
    
    
       % Ls.('l'+string(i)) = li_closedform;
       % assign(eval('l'+string(i)), li_closedform);
       % % assign(eval('l'+string(i)), best_lis(i))
    
    
       % li = eval('l'+string(i)); %l1
       % disp(value(li))
       li = 1/(2*m^2*max(eig(Wi*Inv_M_prev*Wi')));
    
       Xi = li*eye(di)-li^2*m^2*Wi*Inv_M_prev*Wi';
       % Xis_eig_min = [Xis_eig_min min(eig(value(Xi)))];
    
       Xi_prev = Xi;
       % d_cum = d_cum+di;
    end 
    
    Wl = eval('W'+string(l));
    % F = (min(eig(value(Xi_prev))))/max(eig(Wl'*Wl));
    oneoverF = max(eig((Wl'*Wl)/(value(Xi_prev))));
    Lip_sq_est = oneoverF;
    time_used = toc;

    
    % disp('Verification')
    
    Lip_est = sqrt(Lip_sq_est);
    % trival_Lip_sq = (norm(W1)*norm(W2)*norm(W3)*norm(W4)*norm(W5)*norm(W6)*norm(W7)*norm(W8))^2
    trivial_Lip_sq = 1;
    for i = 1:l
        eval(['trivial_Lip_sq = trivial_Lip_sq * norm(W' num2str(i) ')^2;'])
    end

    trivial_Lip = sqrt(trivial_Lip_sq);

end
