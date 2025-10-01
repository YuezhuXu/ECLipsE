from torch import nn

def extract_model_info(model):
    """
    Given a PyTorch model, returns:
    - weights: list of weight tensors
    - sizes: list of weight shapes
    - activations: list of activation function names (or None)
    - num_layers: number of parameterized layers
    """
    weights_list = []
    sizes_list = []
    activations_list = []
    alpha_list = []
    beta_list = []

    modules = list(model.modules())[1:]  # skip the top-level model itself

    for i, layer in enumerate(modules):
        if isinstance(layer, (nn.Linear)):
            # Store weights and shapes
            weights_list.append(layer.weight.data.clone().double())
            sizes_list.append(tuple(layer.weight.shape))
            
            # Look ahead for activation
            act_name = None
            if i + 1 < len(modules):
                next_layer = modules[i + 1]
                if isinstance(next_layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.Softmax)):
                    act_name = next_layer.__class__.__name__
            
            activations_list.append(act_name)
            alpha_list.append(0.0)  # Placeholder for alpha
            beta_list.append(1.0) if act_name != 'Sigmoid' else beta_list.append(0.25)
            
    num_layers = len(weights_list)
    return weights_list, sizes_list, activations_list, alpha_list, beta_list, num_layers