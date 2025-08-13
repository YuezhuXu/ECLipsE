function y = leakyrelu(x, alpha)
    if nargin < 2, alpha = 0.1; end
    y = x;
    y(x < 0) = alpha * x(x < 0);
end
