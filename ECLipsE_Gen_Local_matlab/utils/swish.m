function y = swish(x)
    sig = 1 ./ (1 + exp(-x));
    y = x .* sig;
end
