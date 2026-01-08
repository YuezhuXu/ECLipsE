"""
Quick example: Using the merged LipConstEstimator with estimate_gen_local
"""

import numpy as np
from src.eclipse_nn import LipConstEstimator

# Create estimator with a simple neural network
estimator = LipConstEstimator()
estimator.generate_random_weights([10, 20, 15, 5])

# Define local region for estimation
center = np.random.randn(10)  # 10-dimensional center point
epsilon = 0.1                  # Local region radius

# Compute local Lipschitz constant
Lip, time_used, ext = estimator.estimate_gen_local(
    center=center,
    epsilon=epsilon,
    actv='relu',      # Activation function
    algo='Fast'        # Algorithm variant
)

print(f"Local Lipschitz Constant: {Lip:.6f}")
print(f"Computation Time: {time_used:.4f} seconds")
print(f"Status: {'Success' if ext == 0 else 'Failed'}")

# Compare with global estimation
lip_global = estimator.estimate('ECLipsE_Fast')
print(f"\nGlobal Lipschitz Constant: {lip_global:.6f}")
print(f"Ratio (Local/Global): {Lip/lip_global:.4f}")
