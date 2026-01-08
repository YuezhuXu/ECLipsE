"""
Test script to demonstrate the merged LipConstEstimator class with get_lip_estimates method.
"""
import torch
import numpy as np
from src.eclipse_nn import LipConstEstimator

# Example 1: Using get_lip_estimates with random weights
print("=" * 60)
print("Test 1: Using get_lip_estimates with random weights")
print("=" * 60)

# Create a simple neural network with random weights
layers = [10, 20, 15, 5]
estimator = LipConstEstimator()
estimator.generate_random_weights(layers)

# Define center point and epsilon for local Lipschitz estimation
center = np.random.randn(layers[0])
epsilon = 0.1

# Compute local Lipschitz constant using the new method
print(f"\nNetwork architecture: {layers}")
print(f"Center point dimension: {center.shape[0]}")
print(f"Epsilon (local region radius): {epsilon}")

try:
    Lip, time_used, ext = estimator.estimate_gen_local(
        center=center,
        epsilon=epsilon,
        actv='relu',
        algo='Fast'
    )
    
    print(f"\n✓ Local Lipschitz estimate: {Lip:.6f}")
    print(f"✓ Computation time: {time_used:.4f} seconds")
    print(f"✓ Exit code: {ext} (0=success, -1=failure)")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Example 2: Compare with global methods
print("\n" + "=" * 60)
print("Test 2: Compare with global estimation methods")
print("=" * 60)

# Use existing global methods
try:
    lip_fast = estimator.estimate('ECLipsE_Fast')
    print(f"✓ Global Lipschitz (ECLipsE_Fast): {lip_fast:.6f}")
except Exception as e:
    print(f"✗ ECLipsE_Fast error: {e}")

try:
    lip_eclipse = estimator.estimate('ECLipsE')
    print(f"✓ Global Lipschitz (ECLipsE): {lip_eclipse:.6f}")
except Exception as e:
    print(f"✗ ECLipsE error: {e}")

# Example 3: Test with different activation functions
print("\n" + "=" * 60)
print("Test 3: Testing different activation functions")
print("=" * 60)

activations = ['relu', 'tanh', 'sigmoid']

for actv in activations:
    try:
        Lip, time_used, ext = estimator.estimate_gen_local(
            center=center,
            epsilon=epsilon,
            actv=actv,
            algo='Fast'
        )
        print(f"✓ {actv:10s}: Lip = {Lip:.6f}, time = {time_used:.4f}s")
    except Exception as e:
        print(f"✗ {actv:10s}: Error - {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
