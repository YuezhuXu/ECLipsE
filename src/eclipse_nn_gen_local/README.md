# ECLipsE-Gen-Local: Generalized Local Lipschitz Constant Estimation

This package provides an improved version of the ECLipsE algorithm for estimating Lipschitz constants of neural networks using **local activation bounds**.

## Key Differences from Original ECLipsE

### Original ECLipsE
- Uses **global** activation bounds (e.g., α=0, β=1 for ReLU)
- Assumes the same activation bounds for all neurons in a layer
- Works with weights only (no biases)
- Two variants: standard (SDP-based) and Fast (closed-form)

### ECLipsE-Gen-Local (This Implementation)
- Uses **local** activation bounds computed per-neuron based on input region
- More accurate: computes activation bounds for each neuron individually given a center point and radius
- Three algorithms:
  - **Acc** (Accurate): Uses SDP with per-neuron λ optimization
  - **Fast**: Uses SDP with scalar λ for all neurons
  - **CF** (Closed-Form): Analytical solution (only when α·β ≥ 0)
- Considers both weights AND biases
- Tracks value propagation through the network for tighter local bounds
- Fallback logic: if Acc fails, automatically tries Fast or CF
- Supports multiple activation functions:
  - ReLU, LeakyReLU, ELU
  - Sigmoid, Tanh
  - SiLU/Swish, Softplus

## Installation

The package is part of the `eclipse-nn` PyPI package. Install with:

```bash
pip install eclipse-nn
```

## Usage

```python
import torch
import numpy as np
from eclipse_nn_gen_local import get_lip_estimates

# Define your neural network weights and biases
weights = [
    torch.randn(100, 784, dtype=torch.float64),  # Layer 1
    torch.randn(100, 100, dtype=torch.float64),  # Layer 2
    torch.randn(10, 100, dtype=torch.float64)    # Output layer
]

biases = [
    torch.randn(100, 1, dtype=torch.float64),
    torch.randn(100, 1, dtype=torch.float64),
    torch.randn(10, 1, dtype=torch.float64)
]

# Define local region: center point and radius
center = torch.randn(784, 1, dtype=torch.float64)  # Input dimension
epsilon = 1.0  # Radius of local region

# Compute Lipschitz constant estimate
Lip, time_used, exit_code = get_lip_estimates(
    weights=weights,
    biases=biases,
    actv='relu',      # Activation function
    center=center,    # Center of local region
    epsilon=epsilon,  # Radius of local region
    algo='Fast'       # Algorithm: 'Acc', 'Fast', or 'CF'
)

print(f"Local Lipschitz constant: {Lip}")
print(f"Computation time: {time_used:.4f}s")
print(f"Exit code: {exit_code} (0=success, -1=failure)")
```

## Algorithm Selection Guide

- **Acc (Accurate)**: Most accurate but slowest. Optimizes λ for each neuron independently.
- **Fast**: Good balance of speed and accuracy. Uses a scalar λ for all neurons.
- **CF (Closed-Form)**: Fastest but only works when α·β ≥ 0 (e.g., ReLU, LeakyReLU).

The algorithm will automatically fall back to simpler methods if the chosen one fails.

## Supported Activation Functions

- `'relu'`: Rectified Linear Unit
- `'leakyrelu'`: Leaky ReLU (α=0.01)
- `'elu'`: Exponential Linear Unit
- `'sigmoid'`: Sigmoid
- `'tanh'`: Hyperbolic tangent
- `'silu'`: Sigmoid Linear Unit (SiLU)
- `'swish'`: Swish (same as SiLU)
- `'softplus'`: Softplus

## Module Structure

```
eclipse_nn_gen_local/
├── __init__.py                  # Package initialization
├── get_lip_estimates.py         # Main estimation function
├── find_good_lambdas.py         # Lambda optimization (Acc/Fast/CF)
├── actv_slope_range.py          # Activation slope bounds computation
└── activation_functions.py      # Activation function implementations
```

## Theory

ECLipsE-Gen-Local improves upon the original ECLipsE by computing **local** activation bounds for each neuron based on:
1. A center point `c` in the input space
2. A radius `ε` defining the local region around `c`

For each layer, the algorithm:
1. Propagates the center value and uncertainty through the network
2. Computes per-neuron bounds on the activation function slopes [α_i, β_i]
3. Solves an optimization problem to find the best Λ matrix
4. Updates the metric matrices M_i and X_i for the next layer

This approach provides **tighter** Lipschitz bounds than global methods, especially useful for:
- Local robustness analysis
- Certified defenses around specific inputs
- Verification of neural network properties in local regions

## References

Based on the NeurIPS 2024 paper on ECLipsE and its generalized local extension.

## License

MIT License - See LICENSE file in the root directory.
