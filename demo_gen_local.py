"""
Demo script for ECLipsE-Gen-Local Lipschitz constant estimation.

This script demonstrates how to use the get_lip_estimates function
for computing local Lipschitz constants of neural networks.
"""

import torch
import numpy as np
from eclipse_nn_gen_local import get_lip_estimates


def generate_random_network(layers, seed=42):
    """
    Generate random network weights and biases.
    
    Args:
        layers: List of layer sizes, e.g., [784, 100, 100, 10]
        seed: Random seed for reproducibility
    
    Returns:
        tuple: (weights, biases)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    weights = []
    biases = []
    
    for i in range(1, len(layers)):
        # Weight matrix: output_dim x input_dim
        W = torch.randn(layers[i], layers[i-1], dtype=torch.float64) * 0.1
        # Bias vector: output_dim x 1
        b = torch.randn(layers[i], 1, dtype=torch.float64) * 0.1
        
        weights.append(W)
        biases.append(b)
    
    return weights, biases


def demo_random_network():
    """
    Demo 1: Random neural network with varying architectures and algorithms.
    """
    print("=" * 70)
    print("Demo 1: Random Neural Networks")
    print("=" * 70)
    
    # Network architectures to test
    layer_configs = [
        [10, 20, 20, 10],  # Small network
        [50, 100, 100, 50],  # Medium network
    ]
    
    # Activation function
    actv = 'relu'
    
    # Algorithms to compare
    algos = ['Fast', 'CF', 'Acc']
    
    # Local region parameters
    epsilon = 1.0
    
    for layers in layer_configs:
        print(f"\nNetwork architecture: {layers}")
        
        # Generate random network
        weights, biases = generate_random_network(layers)
        
        # Generate random center point
        center = torch.randn(layers[0], 1, dtype=torch.float64)
        
        # Test each algorithm
        for algo in algos:
            try:
                Lip, time_used, ext = get_lip_estimates(
                    weights=weights,
                    biases=biases,
                    actv=actv,
                    center=center,
                    epsilon=epsilon,
                    algo=algo
                )
                
                if ext == 0:
                    print(f"  {algo:6s}: Lip = {Lip:.6f}, Time = {time_used:.4f}s")
                else:
                    print(f"  {algo:6s}: Failed (exit code {ext})")
            except Exception as e:
                print(f"  {algo:6s}: Error - {str(e)[:50]}")


def demo_varying_radius():
    """
    Demo 2: Observe how Lipschitz constant changes with local region radius.
    """
    print("\n" + "=" * 70)
    print("Demo 2: Varying Radius (Lipschitz Tightness)")
    print("=" * 70)
    
    # Fixed network architecture
    layers = [128, 128, 128, 10]
    actv = 'leakyrelu'
    algo = 'Fast'
    
    # Generate network
    weights, biases = generate_random_network(layers)
    center = torch.randn(layers[0], 1, dtype=torch.float64)
    
    # Test different radii
    radii = [5.0, 1.0, 0.2, 0.04, 0.008, 0.0016, 0.00032]
    
    print(f"\nNetwork: {layers}")
    print(f"Activation: {actv}")
    print(f"Algorithm: {algo}")
    print(f"\n{'Radius':>12s}  {'Lipschitz':>12s}  {'Time (s)':>10s}")
    print("-" * 40)
    
    for epsilon in radii:
        try:
            Lip, time_used, ext = get_lip_estimates(
                weights=weights,
                biases=biases,
                actv=actv,
                center=center,
                epsilon=epsilon,
                algo=algo
            )
            
            if ext == 0:
                print(f"{epsilon:12.5f}  {Lip:12.6f}  {time_used:10.4f}")
            else:
                print(f"{epsilon:12.5f}  {'Failed':>12s}  {time_used:10.4f}")
        except Exception as e:
            print(f"{epsilon:12.5f}  Error: {str(e)[:20]}")


def demo_activation_functions():
    """
    Demo 3: Compare different activation functions.
    """
    print("\n" + "=" * 70)
    print("Demo 3: Different Activation Functions")
    print("=" * 70)
    
    # Fixed network
    layers = [50, 100, 100, 10]
    weights, biases = generate_random_network(layers)
    center = torch.randn(layers[0], 1, dtype=torch.float64)
    epsilon = 1.0
    algo = 'Fast'
    
    # Test different activations
    activations = ['relu', 'leakyrelu', 'sigmoid', 'tanh', 'elu', 'silu', 'softplus']
    
    print(f"\nNetwork: {layers}")
    print(f"Algorithm: {algo}")
    print(f"Epsilon: {epsilon}")
    print(f"\n{'Activation':>12s}  {'Lipschitz':>12s}  {'Time (s)':>10s}")
    print("-" * 40)
    
    for actv in activations:
        try:
            Lip, time_used, ext = get_lip_estimates(
                weights=weights,
                biases=biases,
                actv=actv,
                center=center,
                epsilon=epsilon,
                algo=algo
            )
            
            if ext == 0:
                print(f"{actv:>12s}  {Lip:12.6f}  {time_used:10.4f}")
            else:
                print(f"{actv:>12s}  {'Failed':>12s}  {time_used:10.4f}")
        except Exception as e:
            print(f"{actv:>12s}  Error: {str(e)[:20]}")


def demo_pytorch_model():
    """
    Demo 4: Use with a PyTorch model.
    """
    print("\n" + "=" * 70)
    print("Demo 4: PyTorch Model Integration")
    print("=" * 70)
    
    # Define a simple PyTorch model
    class SimpleNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(28*28, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, 10)
            self.relu = torch.nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Create model and extract weights/biases
    model = SimpleNet()
    model.eval()
    
    weights = []
    biases = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights.append(param.data.double())
        elif 'bias' in name:
            biases.append(param.data.double().reshape(-1, 1))
    
    # Random center point
    center = torch.randn(28*28, 1, dtype=torch.float64)
    epsilon = 1.0
    
    print("\nPyTorch Model: SimpleNet")
    print(f"Architecture: [784, 128, 64, 10]")
    print(f"Activation: ReLU")
    
    for algo in ['Fast', 'CF']:
        try:
            Lip, time_used, ext = get_lip_estimates(
                weights=weights,
                biases=biases,
                actv='relu',
                center=center,
                epsilon=epsilon,
                algo=algo
            )
            
            if ext == 0:
                print(f"\n{algo} Algorithm:")
                print(f"  Local Lipschitz constant: {Lip:.6f}")
                print(f"  Computation time: {time_used:.4f}s")
        except Exception as e:
            print(f"\n{algo} Algorithm: Error - {e}")


def main():
    """
    Run all demos.
    """
    print("\n" + "=" * 70)
    print("ECLipsE-Gen-Local Demo")
    print("Local Lipschitz Constant Estimation for Neural Networks")
    print("=" * 70)
    
    # Run demos
    demo_random_network()
    demo_varying_radius()
    demo_activation_functions()
    demo_pytorch_model()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
