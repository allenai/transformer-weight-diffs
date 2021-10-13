from transformers import T5ForConditionalGeneration
from transformers import T5Config
from torch import linalg as LA
import torch
from collections import defaultdict
import math
from model_constants import MODELS_TO_STORE, MODELS_TO_LAYERS
import argparse
from utils import sim_matrix, process_results, get_norm, get_models
from get_state import get_state_dict

def run_difference_calculation(model_folder="t5-large", model_name="t5-large", function_type="l1"):
    assert model_name in MODELS_TO_STORE, f"{model_name} is not one of the following implemented models: {MODELS_TO_STORE.keys()}"
    
    dirname = model_folder
    results_decoder = defaultdict(list)
    model, config = get_models(model_folder, model_name)
    org_model, org_config = get_models(model_name)

    org_dict = org_model.state_dict()
    trained_dict = model.state_dict()

    if model_name in MODELS_TO_STORE:
        model_name, model_size = model_name.split('-')
        model_dict = MODELS_TO_LAYERS[model_name]
        size = MODELS_TO_LAYERS[model_name][model_size]
    else:
        size, model_dict = get_state_dict(model_name)

    if any(['encoder' in state for state in model_dict]):
        table_file_encoder = open(f'{function_type}_encoder_{model_name}.tsv', 'w')
        results_encoder = defaultdict(list)
        for encoder_n in range(size):
            for dict_key in MODELS_TO_STORE[model_name][encoder]:
                dict_names = MODELS_TO_STORE[model_name][encoder]
                q_org = org_dict[dict_names[dict_key].format(encoder_n)]
                q_new = trained_dict[dict_names[dict_key].format(encoder_n)]
                if function_type == "l1":
                    results_encoder[dict_key].append(get_norm(q_org - q_new, 1))
                if function_type == "cossim":
                    results_encoder[dict_key].append(sim_matrix(v_org, v_new))
        process_results(results_encoder)

        for item in results_encoder:
            st = "\t".join([str(float(f.item())) for f in results_encoder[item]])
            table_file_encoder.write(f'{item}\t{st}\n')
    
    if any(['decoder' in state for state in model_dict]):
        table_file_decoder = open(f'{function_type}_decoder_{model_name}.tsv', 'w')
        results_decoder = defaultdict(list)
        for decoder_n in range(size):
            for dict_key in MODELS_TO_STORE[model_name][encoder]:
                dict_names = MODELS_TO_STORE[model_name][encoder]
                q_org = org_dict[dict_names[dict_key].format(decoder_n)]
                q_new = trained_dict[dict_names[dict_key].format(decoder_n)]
                results_decoder[dict_key].append(get_norm(q_org - q_new, 1))
                if function_type == "l1":
                    results_encoder[dict_key].append(get_norm(q_org - q_new, 1))
                if function_type == "cossim":
                    results_encoder[dict_key].append(sim_matrix(v_org, v_new))
        process_results(results_decoder)

        for item in results_decoder:
            st = "\t".join([str(float(f.item())) for f in results_decoder[item]])
            table_file_decoder.write(f'{item}\t{st}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='t5-large')
    parser.add_argument('--model_folder', default='t5-large')
    parser.add_argument('--function_type', default='l1')
    args = parser.parse_args()

    run_difference_calculation(args.model_folder, args.model_name, args.function_type)