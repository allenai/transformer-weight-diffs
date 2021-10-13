from transformers import AutoModel
import re

def replace_layer(name):
    layer_n = re.search('[0-9]+', name).group()
    return layer_n, name.replace(layer_n, '{}', 1)

def get_state_dict(model):
    state_dict = model.state_dict()
    result = set()
    layers = set()
    for state in state_dict.keys():
        if 'weight' not in state:
            continue
        if 'layer' not in state:
            continue
        if 'norm' in state.lower():
            continue
        if 'bias' in state:
            continue
        layer_n, formatted_state = replace_layer(state)
        result.add(formatted_state)
        layers.add(layer_n)
    return len(layers), list(result)
