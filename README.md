# Calculating Weight Differences in Transformers

![Image of heatmaps in T5](images/heatmap.png)

This repo contains code to calculate weight difference before and after fine-tuning in Transformers (e.g. from HuggingFace).

### Setup

For setup, run `pip install requirements.txt`. This repo requires `python 3.6` or higher.

In addition, you'll need a fine-tuned model file to compare to. For an example, the model uses the pretrained (without extra fine-tuning) T5 model. You can use your own fine-tuned model, or download a COMET model file from `https://github.com/allenai/few-shot-comet`.

### Running Difference Calculation

The following is an example of getting difference calculation running for a T5-COMET model.

(1) Place a model folder in a local directory, for example, in `./fine-tuned-t5`

(2) Run `python calculate_differences.py` with appropriate parameters:

- `python calculate_differences.py --model_name t5-large --model_folder t5-large --function_type l1`

Likely, replace `--model_folder t5` ith the location of your fine-tuned weights.

(3) (Optional) Create a heatmap with the weights produced from the model, for example, using `heatmap/heatmaps.py`

(Runtime) The runtime for calculating weights is, on Quadro RTX 8000: 

`t5-large`: 1m 17s

`t5-11b`: ~37m

### Adding your own model

To add a new model type to this script, you'll need to modify `model_constants.py` with the information for your model.

We also have an automatic function that will attempt to parse the weight dict and fill in the model if the model name is not in `model_constants.py`.

- `MODELS_TO_LAYERS` will need to be updated with the number of layers for the model of the given size.
- `MODELS_TO_STORE` will need to be updated with maps of the model config to the name of the weights stored in the model. An example is provided for T5, which uses both an encoder and a decoder.

### Citation information

If you use this repo for a publication, please cite:

```
@inproceedings{Da2021FEWSHOTKMS,
  title={Analyzing Commonsense Emergence in Few-shot Knowledge Models},
  author={Jeff Da and Ronan Le Bras and Ximing Lu and Yejin Choi and Antoine Bosselut},
  booktitle={AKBC},
  year={2021}
}
```

### Notes

- For public inquiries, please create a GitHub issue with questions and suggestions. We suggest that you create a pull request instead of an issue for fixes and updates.
- Please contact jeffd@allenai.org for any private inquiries.

### License
This work is under the Apache 2.0 license.