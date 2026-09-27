# Fine-Tuning

## Scope

Fine tuning LLM for dialog summarization.

## Lab

Lab: [Lab_2_fine_tune_generative_ai_model.ipynb](Lab_2_fine_tune_generative_ai_model.ipynb)

## Dataset

[load_dataset.py](load_dataset.py)

## Model

This lab is using [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) model from hugging face using the [transformers](https://huggingface.co/docs/transformers/en/index) python package.

Lab: [Lab_2_fine_tune_generative_ai_model.ipynb](Lab_2_fine_tune_generative_ai_model.ipynb)

## Load Model

[load_model.py](load_model.py)

## View Model Parameters

[view_model_params.py](view_model_params.py)

```commandline
output:
trainable model parameters: 247577856
all model parameters: 247577856
percentage of trainable model parameters: 100.00%
```

## Test Zero short Inferencing

[zero_shot_inference_1.py](zero_shot_inference_1.py)

## Fine-Tuning

[full_fine_tuning.py](fine_tune_full.py)

## Human Evaluation

[human_evaluation.py](human_evaluation.py)

## Rouge Evaluation

[rouge_evaluation.py](rouge_evaluation.py)

## PEFT - LORA

[peft_lora_evaluation.py](fine_tune_peft_lora.py)