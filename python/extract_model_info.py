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

    modules = list(model.modules())[1:]  # skip the top-level model itself

    for i, layer in enumerate(modules):
        if isinstance(layer, (nn.Linear)):
            # Store weights and shapes
            weights_list.append(layer.weight.data.clone())
            sizes_list.append(tuple(layer.weight.shape))
            
            # Look ahead for activation
            act_name = None
            if i + 1 < len(modules):
                next_layer = modules[i + 1]
                if isinstance(next_layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.Softmax)):
                    act_name = next_layer.__class__.__name__
            
            activations_list.append(act_name)

    num_layers = len(weights_list)
    return weights_list, sizes_list, activations_list, num_layers