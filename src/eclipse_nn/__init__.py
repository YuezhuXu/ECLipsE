"""
ECLipsE: Efficient Lipschitz Constant Estimation for Neural Networks

This package provides various methods for estimating Lipschitz constants:
- Global estimation: ECLipsE and ECLipsE_Fast
- Local estimation: ECLipsE-Gen-Local
"""

from .LipConstEstimator import LipConstEstimator
from .eclipsE import ECLipsE
from .eclipsE_fast import ECLipsE_Fast
from .local_lipschitz import get_lip_estimates

__version__ = '1.0.0'

__all__ = [
    'LipConstEstimator',
    'ECLipsE',
    'ECLipsE_Fast',
    'get_lip_estimates'
]
