from transformers import AutoModel
import re

"""
Helper methods to automatically create state dictionaries.
"""

"""
Returns the layer id and formatted state name in the given HuggingFace state dictionary name.
name: name of dict from HuggingFace
"""
def replace_layer(name):
    layer_n = re.search('[0-9]+', name).group()
    return layer_n, name.replace(layer_n, '{}', 1)

"""
Automatically gets the state dict from the given HuggingFace model.
model: loaded HuggingFace model
model_name: name of model to store in state dict
"""
def get_state_dict(model, model_name):
    state_dict = model.state_dict()
    state_list = set()
    layers = set()
    encoder_dict = {}
    decoder_dict = {}
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
        state_list.add(formatted_state)
        layers.add(layer_n)
    
    # create state dict
    models_to_store = { model_name: {} }
    if any(['encoder' in state for state in state_list]):
        models_to_store[model_name]["encoder"] = encoder_dict
    if any(['decoder' in state for state in state_list]):
        models_to_store[model_name]["decoder"] = decoder_dict
    
    for state in state_list:
        state_name = state.split('.')[-2]
        if "encoder" in state:
            encoder_dict[state_name] = state
        if "decoder" in state:
            encoder_dict[state_name] = state
    
    layer_dict = {
        model_name: len(layers)
    }

    return layer_dict, models_to_store
