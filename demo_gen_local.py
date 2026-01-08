"""
Demo script for ECLipsE-Gen-Local Lipschitz constant estimation.

"""

import torch
import numpy as np

from src.eclipse_nn import LipConstEstimator


lyr, n = 5, 10
layers = [5] + [n]*(lyr-1) + [2]

center = [0.4, 1.8, -0.5, -1.3, 0.9]
epsilon = 1
activation = 'relu'

np.random.seed(n*77+lyr*9)

# Generate random weights for the neural network
weights = []
biases = []
for i in range(len(layers)-1):
    weights.append(np.random.randn(layers[i+1], layers[i]))
    biases.append(np.random.randn(layers[i+1]))

est = LipConstEstimator(weights=weights, biases=biases)
lip_fast = est.estimate_gen_local(center=center, epsilon=epsilon, actv=activation, algo='Fast')
lip_acc = est.estimate_gen_local(center=center, epsilon=epsilon, actv=activation, algo='Acc')
lip_cf = est.estimate_gen_local(center=center, epsilon=epsilon, actv=activation, algo='CF')
print(f"Local Lipschitz Constant Estimates:")
print(f"ECLipsE-Gen-Local Fast: {lip_fast:.6f}")
print(f"ECLipsE-Gen-Local Accurate: {lip_acc:.6f}")
print(f"ECLipsE-Gen-Local CF: {lip_cf:.6f}")