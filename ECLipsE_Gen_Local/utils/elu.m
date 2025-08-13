function y = elu(x, a)
    if nargin < 2, a = 1.0; end
    y = x;
    y(x < 0) = a * (exp(x(x < 0)) - 1);
end