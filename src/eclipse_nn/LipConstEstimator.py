from .extract_model_info import extract_model_info
from .eclipsE import ECLipsE
from .eclipsE_fast import ECLipsE_Fast
from .local_lipschitz import get_lip_estimates as compute_local_lip
import torch
import numpy as np

class LipConstEstimator():
    def __init__(self, model=None, weights=None, biases=None, alphas=None, betas=None):
        if model is not None and weights is not None:
            raise ValueError("You cannot provide both 'model' and 'weights'. Please provide only one or neither.")
        self.model = model
        self.lipConstE = None
        self.lipConstEfast = None
        self.lipConstTrivial = None

        # Extract model info if model is provided
        if model is not None:
            extracted_weights, extracted_biases, sizes, activations, alphas, betas, num_layers = extract_model_info(model)
            self.weights = extracted_weights
            self.biases = extracted_biases
            self.sizes = sizes
            self.activations = activations
            self.alphas = alphas
            self.betas = betas
            self.num_layers = num_layers
        elif weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
            self.sizes = None
            self.activations = None
            self.alphas = alphas if alphas is not None else [0] * (len(weights)-1)
            self.betas = betas if betas is not None else [1] * (len(weights)-1)
            self.num_layers = len(weights)
        elif weights is not None:
            self.weights = weights
            self.biases = [None] * len(weights)
            self.sizes = None
            self.activations = None
            self.alphas = alphas if alphas is not None else [0] * (len(weights)-1)
            self.betas = betas if betas is not None else [1] * (len(weights)-1)
            self.num_layers = len(weights)
        else:
            self.weights = None
            self.biases = None
            self.sizes = None
            self.activations = None
            self.alphas = None
            self.betas = None
            self.num_layers = 0

    def model_review(self):
        if self.model == None:
            print('No models detected. Please set a model.')
        else:
            print('MODEL INFO')
            for i_layer in range(self.num_layers):
                print(f'Layer #{i_layer}: input size = {self.sizes[i_layer][1]}, output size = {self.sizes[i_layer][0]}, activation = {self.activations[i_layer]}.')
            print('REMARK: ONLY FULLY CONNECT NEURAL NETS ARE APPLICABLE IN THIS ESTIMATOR.')

    def generate_random_weights(self, layers):
        self.weights = []
        for l in range(1, len(layers)):
            self.weights.append(torch.rand([layers[l], layers[l-1]], dtype=torch.float64))
        self.num_layers = len(self.weights)
        self.alphas = [0] * (self.num_layers - 1)
        self.betas = [1] * (self.num_layers - 1)

    def estimate(self, method):
        if method == 'trivial':
            # trivial = 1
            # for i in range(len(self.weights)):
            #     trivial *= torch.linalg.norm(self.weights[i])**2
            # return torch.sqrt(trivial)
            return np.sqrt(np.prod(list([torch.linalg.norm(self.weights[i])**2] for i in range(len(self.weights)))))
        elif method == 'ECLipsE':
            return ECLipsE(self.weights, self.alphas, self.betas)
        elif method == 'ECLipsE_Fast':
            return ECLipsE_Fast(self.weights, self.alphas, self.betas)
        else:
            print('INVALID METHOD')

    def estimate_gen_local(self, center, epsilon, actv='relu', algo='Fast'):
        """
        Compute local Lipschitz constant estimate for a neural network using ECLipsE-Gen-Local.
        
        Args:
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
        return compute_local_lip(self.weights, center, epsilon, actv, self.biases, algo)
