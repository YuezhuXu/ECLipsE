function [slope_low, slope_high] = ActvSlopeRange(actType, z_ranges)
% ActvSlopeRange   Strict analytic slope bounds for activations
%
% [slope_low, slope_high] = ActvSlopeRange(actType, z_ranges)
%   actType    = 'relu' | 'sigmoid' | 'tanh' | 'leakyrelu' | 'elu' | 'swish' | 'softplus'
%   z_ranges   = N×2 array, each row = [z_min, z_max]
%
% Outputs:
%   slope_low  = N×1 vector of min f'(z) over each interval
%   slope_high = N×1 vector of max f'(z) over each interval

N = size(z_ranges,1);
slope_low  = zeros(N,1);
slope_high = zeros(N,1);

alpha = 0.01; % LeakyReLU param
elu_a = 1.0;  % ELU param

for i = 1:N
    z_min = z_ranges(i,1);
    z_max = z_ranges(i,2);

    switch lower(actType)
        case 'relu'
            slope_low(i)  = double(z_min >= 0);
            slope_high(i) = double(z_max > 0);

        case 'leakyrelu'
            slope_low(i)  = (z_max <= 0)*alpha + (z_min >= 0)*1 + (z_min<0 && z_max>0)*alpha;
            slope_high(i) = (z_max <= 0)*alpha + (z_min >= 0)*1 + (z_min<0 && z_max>0)*1;

        case 'sigmoid'
            sig_prime = @(z) exp(-z)./(1+exp(-z)).^2;
            slope_low(i)  = min(sig_prime(z_min), sig_prime(z_max));
            if z_min <= 0 && z_max >= 0
                slope_high(i) = 1/4;
            else
                slope_high(i) = max(sig_prime(z_min), sig_prime(z_max));
            end

        case 'tanh'
            tanh_prime = @(z) 1 - tanh(z).^2;
            slope_low(i)  = min(tanh_prime(z_min), tanh_prime(z_max));
            if z_min <= 0 && z_max >= 0
                slope_high(i) = 1;
            else
                slope_high(i) = max(tanh_prime(z_min), tanh_prime(z_max));
            end

        case 'elu'
            if z_max <= 0
                slope_low(i)  = elu_a*exp(z_min);
                slope_high(i) = elu_a*exp(z_max);
            elseif z_min >= 0
                slope_low(i)  = 1;
                slope_high(i) = 1;
            else
                slope_low(i)  = elu_a*exp(z_min);
                slope_high(i) = 1;
            end
        case 'silu'
            % SiLU: x * sigmoid(x)
            sigmoid = @(x) 1 ./ (1 + exp(-x));
            
            % SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            silu_deriv = @(x) sigmoid(x) .* (1 + x .* (1 - sigmoid(x)));
            
            if z_max <= z_min % degenerate/point interval
                s = silu_deriv(z_min);
                slope_low(i) = s;
                slope_high(i) = s;
            else
                % Evaluate endpoints
                dmin = silu_deriv(z_min);
                dmax = silu_deriv(z_max);
                
                % Find critical points by solving silu_deriv'(x) = 0
                % Second derivative of SiLU:
                silu_deriv2 = @(x) sigmoid(x) .* (1 - sigmoid(x)) .* (2 + x .* (1 - 2.*sigmoid(x)));
                
                % Find critical points in the interior
                critical_points = [];
                
                % Use multiple starting points to find all critical points
                n_starts = 10;
                start_points = linspace(z_min, z_max, n_starts);
                
                opts = optimset('Display','off','TolX',1e-12,'TolFun',1e-12);
                
                for j = 1:length(start_points)
                    try
                        % Find where second derivative is zero (critical points of first derivative)
                        cp = fzero(silu_deriv2, start_points(j), opts);
                        if cp >= z_min && cp <= z_max
                            % Check if this is a new critical point
                            if isempty(critical_points) || all(abs(critical_points - cp) > 1e-10)
                                critical_points = [critical_points, cp];
                            end
                        end
                    catch
                        % fzero failed, continue
                    end
                end
                
                % Evaluate derivative at all critical points
                all_points = [z_min, z_max];
                if ~isempty(critical_points)
                    all_points = [all_points, critical_points];
                end
                
                all_values = silu_deriv(all_points);
                
                slope_low(i) = min(all_values);
                slope_high(i) = max(all_values);
            end
    
            case 'swish'
                sig = @(z) 1./(1+exp(-z));
                swish_prime = @(z) sig(z) + z.*sig(z).*(1 - sig(z));
    
                slope_low(i)  = min(swish_prime(z_min), swish_prime(z_max));
                if (z_min <= 1.2785) && (z_max >= 1.2785)
                    slope_high(i) = 1.1; % known global maximum at z~1.2785
                else
                    slope_high(i) = max(swish_prime(z_min), swish_prime(z_max));
                end

        case 'softplus'
            sig = @(z) 1./(1+exp(-z));
            slope_low(i) = min(sig(z_min), sig(z_max));
            slope_high(i) = max(sig(z_min), sig(z_max));

        otherwise
            error('Unknown activation "%s".', actType)
    end
end
end
