import os
# Prevent OpenMP/MKL fights between torch/numpy/scs on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch.nn as nn
import numpy as np
import torch
from eclipse_nn.LipConstEstimator import LipConstEstimator

'''
    create estimator by torch model
'''

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.act1(self.fc1(x))
        return self.act2(self.fc2(x))

model = SimpleNet()
est = LipConstEstimator(model=model)
# est.model_review()
print(f'Number of layers: {est.num_layers}')
print(f'alphas: {est.alphas}')
print(f'betas: {est.betas}')
print(f'activations: {est.activations}')
lip_trivial = est.estimate(method='trivial')
lip_fast = est.estimate(method='ECLipsE')
print(f'Trivial Lip Const = {lip_trivial}')
print(f'EclipsE Fast Lip Const = {lip_fast}')
print(f'Ratio = {lip_fast / lip_trivial}')


'''
    create estimator by given weights
'''
print('=================================')
weights_npz = np.load('sampleweights' + os.sep + 'npz' + os.sep + 'lyr' + str(2) + 'n' + str(80) + 'test' + str(1) + '.npz')
weights = []
for i in range(1,2+1):
    weights.append(torch.tensor(weights_npz['w'+str(i)]))
est = LipConstEstimator(weights=weights)
print('Default values for alphas and betas are 0 and 1 vectors respectively.')
print(f'alphas: {est.alphas}')
print(f'betas: {est.betas}')
lip_trivial = est.estimate(method='trivial')
lip_fast = est.estimate(method='ECLipsE_Fast')
print(f'Trivial Lip Const = {lip_trivial}')
print(f'EclipsE Fast Lip Const = {lip_fast}')
print(f'Ratio = {lip_fast / lip_trivial}')
print('Change betas to 0.25')
est.betas = [0.25] * (est.num_layers - 1)
lip_fast = est.estimate(method='ECLipsE_Fast')
print(f'EclipsE Fast Lip Const (after change) = {lip_fast}')


'''
    create estimator by nothing
'''
print('=================================')
est = LipConstEstimator()
est.generate_random_weights([10,20,3])
lip_trivial = est.estimate(method='trivial')
lip_fast = est.estimate(method='ECLipsE_Fast')
print(f'Trivial Lip Const = {lip_trivial}')
print(f'EclipsE Fast Lip Const = {lip_fast}')
print(f'Ratio = {lip_fast / lip_trivial}')


'''
    Example for exceptions
'''
print('=================================')
est = LipConstEstimator(model=model, weights=weights)  # should raise ValueError
