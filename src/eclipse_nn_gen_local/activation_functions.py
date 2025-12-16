"""
Activation functions for ECLipsE-Gen-Local
"""
import torch
import numpy as np


def elu(x, alpha=1.0):
    """
    Exponential Linear Unit (ELU) activation function.
    
    Args:
        x: Input tensor or array
        alpha: Scaling parameter for negative values (default: 1.0)
    
    Returns:
        ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
    """
    if isinstance(x, torch.Tensor):
        return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    else:
        x = np.asarray(x)
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def leakyrelu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    
    Args:
        x: Input tensor or array
        alpha: Slope for negative values (default: 0.01)
    
    Returns:
        LeakyReLU(x) = x if x > 0, alpha * x if x <= 0
    """
    if isinstance(x, torch.Tensor):
        return torch.where(x > 0, x, alpha * x)
    else:
        x = np.asarray(x)
        return np.where(x > 0, x, alpha * x)


def sigmoid(x):
    """
    Sigmoid activation function.
    
    Args:
        x: Input tensor or array
    
    Returns:
        sigmoid(x) = 1 / (1 + exp(-x))
    """
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    else:
        x = np.asarray(x)
        return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    """
    SiLU (Sigmoid Linear Unit) activation function, also known as Swish.
    
    Args:
        x: Input tensor or array
    
    Returns:
        SiLU(x) = x * sigmoid(x)
    """
    if isinstance(x, torch.Tensor):
        return x * torch.sigmoid(x)
    else:
        x = np.asarray(x)
        return x * sigmoid(x)


def swish(x):
    """
    Swish activation function (same as SiLU).
    
    Args:
        x: Input tensor or array
    
    Returns:
        Swish(x) = x * sigmoid(x)
    """
    return silu(x)


def softplus(x):
    """
    Softplus activation function.
    
    Args:
        x: Input tensor or array
    
    Returns:
        Softplus(x) = log(1 + exp(x))
    """
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.softplus(x)
    else:
        x = np.asarray(x)
        return np.log(1.0 + np.exp(x))
