from transformers import T5ForConditionalGeneration
from transformers import T5Config
from transformers import T5ForConditionalGeneration
from transformers import T5Config
from torch import linalg as LA
import torch
from collections import defaultdict
import math

def sim_matrix(a, b, eps=1e-10):
    total = 0
    for dim in range(a.shape[0]):
        cos = torch.nn.CosineSimilarity(dim=0, eps=eps)
        total += math.acos(min(float(cos(a[dim], b[dim]).item()), 1)) / 3.141
    return total / a.shape[0]

def process_results(store):
    pass
    
def get_norm(mat, n=None):
    if n == None:
        return torch.tensor(LA.norm(mat).item())
    return torch.tensor(LA.norm(mat, n).item())

def get_models(model_folder, model_name):
    if model_name == 't5':
        config = T5Config.from_pretrained(model_folder)
        model = T5ForConditionalGeneration.from_pretrained(model_folder, config=config)
        return model, config