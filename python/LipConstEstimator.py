from extract_model_info import extract_model_info

class LipConstEstimator():
    def __init__(self, model=None, weights=None):
        self.model = model
        self.weights = weights
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
            print('Remark: only fully connected layers are counted in this estimator.')
            
import torch.nn as nn
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
est = LipConstEstimator(model)
est.model_review()
print(est.weights[0])
print(type(est.weights[0]))


    