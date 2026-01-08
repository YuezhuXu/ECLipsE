"""
Local Lipschitz constant estimation using ECLipsE-Gen-Local algorithm
"""
import torch
import numpy as np
import time

from .utils.find_good_lambdas import find_good_lambdas
from .utils.actv_slope_range import actv_slope_range
from .utils.activation_functions import elu, leakyrelu, sigmoid, silu, swish, softplus


def get_activation_function(actv_type):
    """
    Get the activation function based on the activation type string.
    
    Args:
        actv_type: String specifying activation type
    
    Returns:
        Callable activation function
    """
    actv_lower = actv_type.lower()
    
    if actv_lower == 'relu':
        return lambda x: np.maximum(0, x)
    elif actv_lower == 'leakyrelu':
        return lambda x: leakyrelu(x, alpha=0.01)
    elif actv_lower == 'sigmoid':
        return sigmoid
    elif actv_lower == 'tanh':
        return np.tanh
    elif actv_lower == 'elu':
        return lambda x: elu(x, alpha=1.0)
    elif actv_lower == 'silu':
        return silu
    elif actv_lower == 'swish':
        return swish
    elif actv_lower == 'softplus':
        return softplus
    else:
        raise ValueError(f'Unknown activation: {actv_type}')


def get_lip_estimates(weights, center, epsilon, actv='relu', biases=None, algo='Fast'):
    """
    Compute local Lipschitz constant estimate for a neural network using ECLipsE-Gen-Local.
    
    Args:
        weights: List of weight matrices/tensors
        center: Center point for local region (d0 x 1 or flat array)
        epsilon: Radius of local region
        actv: Activation function name ('relu', 'leakyrelu', 'sigmoid', 'tanh', 'elu', 'silu', 'swish', 'softplus')
        biases: List of bias vectors (each is di x 1). If None, assumes zero biases.
        algo: Algorithm to use ('Acc', 'Fast', or 'CF')
    
    Returns:
        tuple: (Lip, time_used, ext)
            - Lip: Lipschitz constant estimate
            - time_used: Computation time
            - ext: Exit code (0 = success, -1 = failure)
    """
    if weights is None:
        raise ValueError("No weights available. Please provide a model or weights first.")
    
    # Convert inputs to numpy
    if isinstance(center, torch.Tensor):
        center = center.detach().cpu().numpy()
    center = np.asarray(center).reshape(-1, 1)
    
    weights_np = []
    for w in weights:
        if isinstance(w, torch.Tensor):
            weights_np.append(w.detach().cpu().numpy())
        else:
            weights_np.append(np.asarray(w))
    
    # Handle biases
    if biases is None:
        biases_np = [np.zeros((w.shape[0], 1)) for w in weights_np]
    else:
        biases_np = []
        for b in biases:
            if isinstance(b, torch.Tensor):
                biases_np.append(b.detach().cpu().numpy().reshape(-1, 1))
            else:
                biases_np.append(np.asarray(b).reshape(-1, 1))
    
    N = len(weights_np)
    
    # Compute delta_z_norm
    delta_z_norm2 = epsilon * np.sqrt(center.shape[0])
    
    d0 = weights_np[0].shape[1]
    Mi = np.eye(d0)  # M0
    value_c = center.copy()
    
    start_time = time.time()
    
    ext = 0
    skip = 0
    
    alphas_list = []
    betas_list = []
    
    # Get activation function
    actv_func = get_activation_function(actv)
    
    for i in range(N - 1):
        print(f'Layer {i}')
        
        if skip == 1:
            Wiprev = Wi
            assert np.all(np.abs(alphai - betai) < 1e-20), "Expected affine layer"
        
        Wi = weights_np[i]
        Wi_ori = Wi.copy()
        
        if skip == 1:
            Wi = Wi @ np.diag(alphai.flatten()) @ Wiprev
        
        Winext = weights_np[i + 1]
        bi = biases_np[i]
        
        # Calculate Lipschitz constant for each neuron
        Miprev = Mi
        Ai = np.linalg.pinv(Miprev) @ Wi.T
        Li = np.sqrt(np.sum(Wi * Ai.T, axis=1))
        
        # Update center value
        value_c = Wi_ori @ value_c + bi
        
        # Compute f_range for each neuron
        f_range = np.column_stack([value_c.flatten() - delta_z_norm2 * Li,
                                     value_c.flatten() + delta_z_norm2 * Li])
        
        # Get activation slope ranges
        alphai, betai = actv_slope_range(actv, f_range)
        alphas_list.append(alphai)
        betas_list.append(betai)
        
        # Apply activation function for next iteration
        if i < N - 1:
            value_c = actv_func(value_c)
        
        # CF algorithm check
        if algo == 'CF' and np.any((alphai * betai >= 0) == False):
            raise ValueError('Alphai and Betai do not have matching signs elementwise, CF algorithm not applicable')
        
        # Refine bounds for CF
        if algo == 'CF':
            alphai = np.where(alphai >= 0, 0, alphai)
            betai = np.where(betai <= 0, 0, betai)
        
        # Find good Lambdas
        intv = betai - alphai
        intv_sum = np.sum(intv)
        
        Lambdai, ci, status, Xiprev, Mi = find_good_lambdas(Wi, Winext, Miprev, alphai, betai, algo)
        
        # Handle failures
        if status == 'Failed':
            print("Failed in initial try.")
            skip = 0
            
            if algo == 'Fast':
                alphai = np.where(alphai >= 0, 0, alphai)
                betai = np.where(betai <= 0, 0, betai)
                Lambdai, ci, status, Xiprev, Mi = find_good_lambdas(Wi, Winext, Miprev, alphai, betai, 'CF')
            
            elif algo == 'Acc':
                print('Reduce to Fast/CF.')
                
                # Try Fast
                Lambdai_fast, ci_fast, status_fast, Xiprev_fast, Mi_fast = find_good_lambdas(Wi, Winext, Miprev, alphai, betai, 'Fast')
                
                # Try CF
                alphai_CF = alphai.copy()
                betai_CF = betai.copy()
                alphai_CF = np.where(alphai_CF >= 0, 0, alphai_CF)
                betai_CF = np.where(betai_CF <= 0, 0, betai_CF)
                
                Lambdai_CF, ci_CF, status_CF, Xiprev_CF, Mi_CF = find_good_lambdas(Wi, Winext, Miprev, alphai_CF, betai_CF, 'CF')
                
                if status_fast == 'Failed' or (ci_CF >= ci_fast):
                    print('CF is picked.')
                    Lambdai = Lambdai_CF
                    alphai = alphai_CF
                    betai = betai_CF
                    Xiprev = Xiprev_CF
                    Mi = Mi_CF
                else:
                    print('Fast is picked.')
                    Lambdai = Lambdai_fast
                    Xiprev = Xiprev_fast
                    Mi = Mi_fast
        
        elif status == 'Solved':
            skip = 0
            print('All good.')
        
        elif status == 'Skip':
            print("Affine layer. Directly merge with the next layer.")
            skip = 1
            continue
        
        # Check Xiprev
        if Xiprev is not None and np.min(np.linalg.eigvals(Xiprev)) < 0:
            ext = -1
            print("Xi < 0. Break.")
            break
        
        # Check Mi
        if Mi is not None:
            print(f"Checking Mi: min eigenvalue = {np.min(np.linalg.eigvals(Mi))}")
    
    # Final computation
    WM = weights_np[-1]
    if skip == 1:
        Wiprev = Wi
        assert np.all(np.abs(alphai - betai) < 1e-20), "Expected affine layer"
        WM = WM @ np.diag(alphai.flatten()) @ Wiprev
    
    XMprev = Mi
    fn = WM @ np.linalg.pinv(XMprev) @ WM.T
    fn = (fn + fn.T) / 2
    Lip = np.sqrt(np.max(np.linalg.eigvals(fn)))
    
    time_used = time.time() - start_time
    
    return Lip
