from python.extract_model_info import extract_model_info
from python.ECLipsE import ECLipsE
from python.ECLipsE_Fast import ECLipsE_Fast
import torch
import numpy as np

class LipConstEstimator():
    def __init__(self, model=None, weights=None, alphas=None, betas=None):
        self.model = model
        self.weights = weights
        self.alphas = alphas
        self.betas = betas
        self.lipConstE = None
        self.lipConstEfast = None
        self.lipConstTrivial = None

    def model_review(self):
        if self.model == None:
            print('No models detected. Please set a model.')
        else:
            weights, sizes, activations, num_layers = extract_model_info(self.model)
            self.weights = weights
            self.sizes = sizes
            self.activations = activations
            self.num_layers = num_layers
            print('MODEL INFO')
            for i_layer in range(num_layers):
                print(f'Layer #{i_layer}: input size = {sizes[i_layer][1]}, output size = {sizes[i_layer][0]}, activation = {activations[i_layer]}.')
            print('REMARK: ONLY FULLY CONNECT NEURAL NETS ARE APPLICABLE IN THIS ESTIMATOR.')

    def generate_random_weights(self, layers):
        self.weights = []
        for l in range(1, len(layers)):
            self.weights.append(torch.rand([layers[l], layers[l-1]], dtype=torch.float64))

    def estimate(self, method):
        if method == 'trivial':
            # trivial = 1
            # for i in range(len(self.weights)):
            #     trivial *= torch.linalg.norm(self.weights[i])**2
            # return torch.sqrt(trivial)
            return np.sqrt(np.prod(list([torch.linalg.norm(self.weights[i])**2] for i in range(len(self.weights)))))
        elif method == 'ECLipsE':
            return ECLipsE(self.weights, [0.0]*len(self.weights), [1.0]*len(self.weights))
        elif method == 'ECLipsE_Fast':
            return ECLipsE_Fast(self.weights, [0.0]*len(self.weights), [1.0]*len(self.weights))
        else:
            print('INVALID METHOD')



    