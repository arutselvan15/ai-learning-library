# Lab

## Scope

Feed the conversation between two person to LLM and ask for summary.

Lab: [Lab_1_summarize_dialogue.ipynb](Lab_1_summarize_dialogue.ipynb)

## Data set

Hugging face have lot to data sets available, in this case we are using [knkarthick/dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum).

## Model

This lab is using [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) model from hugging face using the [transformers](https://huggingface.co/docs/transformers/en/index) python package.

## Infra validation

Infra validation checks for specific profile in AWS.

Script: [verify_instance.py](verify_instance.py)

## Load Data Set

Load the data set from hugging face and print the couple of dialogs from the test data.

Script: [load_dataset.py](load_dataset.py)

## Tokenizer

- Load the model
- Send a text to encode function and get the embeddings.
- Send the embedding to decode function and validate it returns same text.

Script: [tokenizer.py](tokenizer.py)

## Summarize Dialog

Starting with basic and improving the [prompts](../../prompt/README.md) with in-context prompt technique.

### Zero shot inference

Pass the dialog to LLM and get the summary.

Script: [zero_shot_inference_1.py](zero_shot_inference_1.py)


Modify the prompt and get the summary.


Script: [zero_shot_inference_2.py](zero_shot_inference_2.py)


### One shot inference

Add one example in the prompt.

Script: [one_shot_inference.py](one_shot_inference.py)


### Few shot inference

Add multiple examples in the prompt.

Script: [few_shot_inference.py](few_shot_inference.py)


### Generative config

Add generative config in the request.

Script: [generative_config_params.py](generative_config_params.py)