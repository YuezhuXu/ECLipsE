"""
Compute activation function slope ranges (alpha and beta bounds)
"""
import numpy as np
from scipy.optimize import fminbound


def actv_slope_range_one_global(actv_type):
    """
    Get global slope bounds for an activation function type.
    
    Args:
        actv_type: String specifying activation type
                   ('relu', 'leakyrelu', 'sigmoid', 'tanh', 'elu', 'silu', 'swish', 'softplus')
    
    Returns:
        tuple: (global_low, global_high) - global min and max derivatives
    """
    alpha_leaky = 0.01
    elu_a = 1.0
    
    actv_type_lower = actv_type.lower()
    
    if actv_type_lower == 'relu':
        return 0.0, 1.0
    elif actv_type_lower == 'leakyrelu':
        return alpha_leaky, 1.0
    elif actv_type_lower == 'sigmoid':
        return 0.0, 0.25
    elif actv_type_lower == 'tanh':
        return 0.0, 1.0
    elif actv_type_lower == 'elu':
        return 0.0, elu_a
    elif actv_type_lower == 'silu':
        return -0.0734, 1.1
    elif actv_type_lower == 'swish':
        return 0.0, 1.1
    elif actv_type_lower == 'softplus':
        return 0.0, 1.0
    else:
        raise ValueError(f'Unknown activation type: {actv_type}')


def actv_slope_range(actv_type, z_ranges):
    """
    Compute strict analytic slope bounds for activation functions over given intervals.
    
    Args:
        actv_type: String specifying activation type
        z_ranges: Nx2 numpy array, each row = [z_min, z_max]
    
    Returns:
        tuple: (slope_low, slope_high) - Nx1 arrays of min/max derivatives over each interval
    """
    N = z_ranges.shape[0]
    slope_low = np.zeros(N)
    slope_high = np.zeros(N)
    
    alpha_leaky = 0.01
    elu_a = 1.0
    
    actv_type_lower = actv_type.lower()
    
    for i in range(N):
        z_min = z_ranges[i, 0]
        z_max = z_ranges[i, 1]
        
        if actv_type_lower == 'relu':
            slope_low[i] = float(z_min >= 0)
            slope_high[i] = float(z_max > 0)
            
        elif actv_type_lower == 'leakyrelu':
            if z_max <= 0:
                slope_low[i] = alpha_leaky
                slope_high[i] = alpha_leaky
            elif z_min >= 0:
                slope_low[i] = 1.0
                slope_high[i] = 1.0
            else:  # z_min < 0 and z_max > 0
                slope_low[i] = alpha_leaky
                slope_high[i] = 1.0
                
        elif actv_type_lower == 'sigmoid':
            def sig_prime(z):
                exp_neg_z = np.exp(-z)
                return exp_neg_z / (1 + exp_neg_z)**2
            
            slope_low[i] = min(sig_prime(z_min), sig_prime(z_max))
            if z_min <= 0 and z_max >= 0:
                slope_high[i] = 0.25
            else:
                slope_high[i] = max(sig_prime(z_min), sig_prime(z_max))
                
        elif actv_type_lower == 'tanh':
            def tanh_prime(z):
                return 1 - np.tanh(z)**2
            
            slope_low[i] = min(tanh_prime(z_min), tanh_prime(z_max))
            if z_min <= 0 and z_max >= 0:
                slope_high[i] = 1.0
            else:
                slope_high[i] = max(tanh_prime(z_min), tanh_prime(z_max))
                
        elif actv_type_lower == 'elu':
            if z_max <= 0:
                slope_low[i] = elu_a * np.exp(z_min)
                slope_high[i] = elu_a * np.exp(z_max)
            elif z_min >= 0:
                slope_low[i] = 1.0
                slope_high[i] = 1.0
            else:
                slope_low[i] = elu_a * np.exp(z_min)
                slope_high[i] = 1.0
                
        elif actv_type_lower == 'silu':
            def sigmoid(x):
                return 1.0 / (1.0 + np.exp(-x))
            
            def silu_deriv(x):
                sig_x = sigmoid(x)
                return sig_x * (1 + x * (1 - sig_x))
            
            if abs(z_max - z_min) < 1e-12:
                s = silu_deriv(z_min)
                slope_low[i] = s
                slope_high[i] = s
            else:
                # Evaluate at endpoints
                vals = [silu_deriv(z_min), silu_deriv(z_max)]
                
                # Find critical points using numerical optimization
                def neg_silu_deriv(x):
                    return -silu_deriv(x)
                
                try:
                    # Find maximum
                    x_max = fminbound(neg_silu_deriv, z_min, z_max)
                    if z_min <= x_max <= z_max:
                        vals.append(silu_deriv(x_max))
                except:
                    pass
                
                try:
                    # Find minimum
                    x_min = fminbound(silu_deriv, z_min, z_max)
                    if z_min <= x_min <= z_max:
                        vals.append(silu_deriv(x_min))
                except:
                    pass
                
                slope_low[i] = min(vals)
                slope_high[i] = max(vals)
                
        elif actv_type_lower == 'swish':
            def sigmoid(z):
                return 1.0 / (1.0 + np.exp(-z))
            
            def swish_prime(z):
                sig_z = sigmoid(z)
                return sig_z + z * sig_z * (1 - sig_z)
            
            slope_low[i] = min(swish_prime(z_min), swish_prime(z_max))
            if z_min <= 1.2785 and z_max >= 1.2785:
                slope_high[i] = 1.1
            else:
                slope_high[i] = max(swish_prime(z_min), swish_prime(z_max))
                
        elif actv_type_lower == 'softplus':
            def sigmoid(z):
                return 1.0 / (1.0 + np.exp(-z))
            
            slope_low[i] = min(sigmoid(z_min), sigmoid(z_max))
            slope_high[i] = max(sigmoid(z_min), sigmoid(z_max))
            
        else:
            raise ValueError(f'Unknown activation type: {actv_type}')
    
    return slope_low, slope_high
