function y = softplus(x)
    % softplus   Numerically stable softplus activation
    %   y = softplus(x)
    %   Computes log(1 + exp(x)), elementwise, for scalar or array x
   y = log1p(exp(-abs(x))) + max(x,0);  % stable version
end