from transformers import T5Config, AutoModel, AutoConfig, T5ForConditionalGeneration
from torch import linalg as LA
import torch
from collections import defaultdict
import math

"""
File containing utilities to calculate weight differences.
"""

"""
Calculates similarity between matrix a and b, rounding to the given eps.
"""
def sim_matrix(a, b, eps=1e-10):
    total = 0
    for dim in range(a.shape[0]):
        cos = torch.nn.CosineSimilarity(dim=0, eps=eps)
        total += math.acos(min(float(cos(a[dim], b[dim]).item()), 1)) / 3.141
    return total / a.shape[0]

"""
Gets the L1 norm of the given matrix.
"""
def get_norm(mat, n=None):
    if n == None:
        return torch.tensor(LA.norm(mat).item())
    return torch.tensor(LA.norm(mat, n).item())

"""
Gets model file and model details from model_folder.
model_folder: fine-tuned model location
model_name: name of model in HuggingFace (e.g. t5-large)
"""
def get_models(model_folder, model_name=None):
    if model_name == 't5-large':
        config = T5Config.from_pretrained(model_folder)
        model = T5ForConditionalGeneration.from_pretrained(model_folder, config=config)
        return model, config
    else:
        config = AutoConfig.from_pretrained(model_folder)
        model = AutoModel.from_pretrained(model_folder, config=config)
        return model, config