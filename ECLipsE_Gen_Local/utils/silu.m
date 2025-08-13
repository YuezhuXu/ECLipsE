function y = silu(x)
    %SILU  Sigmoid Linear Unit activation
    %   y = silu(x) computes y = x .* sigmoid(x)
    %
    %   SiLU(x) = x /    (1 + exp(-x))

    y = x ./ (1 + exp(-x));
end