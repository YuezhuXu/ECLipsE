"""
ECLipsE-Gen-Local: Generalized Local Lipschitz Constant Estimation

This package provides improved Lipschitz constant estimation for neural networks
using local activation bounds computed per-neuron based on input regions.
"""

from .get_lip_estimates import get_lip_estimates
from .find_good_lambdas import find_good_lambdas
from .actv_slope_range import actv_slope_range, actv_slope_range_one_global
from .activation_functions import elu, leakyrelu, sigmoid, silu, swish, softplus

__all__ = [
    'get_lip_estimates',
    'find_good_lambdas',
    'actv_slope_range',
    'actv_slope_range_one_global',
    'elu',
    'leakyrelu',
    'sigmoid',
    'silu',
    'swish',
    'softplus'
]
