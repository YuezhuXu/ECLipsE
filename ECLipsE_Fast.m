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
    
       
       Xi_prev = value(Xi_prev);
       Inv_Xi_prev = inv(Xi_prev);  
    
       
       li = 1/(2*m^2*max(eig(Wi*Inv_Xi_prev*Wi')));
    
       Xi = li*eye(di)-li^2*m^2*Wi*Inv_Xi_prev*Wi';

    
       Xi_prev = Xi;
       % d_cum = d_cum+di;
    end 
    
    Wl = eval('W'+string(l));
    oneoverF = max(eig((Wl'*Wl)/(value(Xi_prev))));
    Lip_sq_est = oneoverF;
    time_used = toc;

    
    
    Lip_est = sqrt(Lip_sq_est);
    trivial_Lip_sq = 1;
    for i = 1:l
        eval(['trivial_Lip_sq = trivial_Lip_sq * norm(W' num2str(i) ')^2;'])
    end

    trivial_Lip = sqrt(trivial_Lip_sq);

end

