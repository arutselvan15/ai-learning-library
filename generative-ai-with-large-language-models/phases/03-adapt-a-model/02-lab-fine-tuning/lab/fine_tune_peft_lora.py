import truststore
import torch
import time
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

# Use macOS system keychain (includes corporate proxy CAs like Cisco Secure Access).
# certifi alone only has public CAs and fails behind SSL inspection.
truststore.inject_into_ssl()

from datasets import load_dataset

tokenizer = None
dash_line = '-'.join('' for x in range(100))


def show_dataset_index(dataset, indexes):

    for i, index in enumerate(indexes):
        print(dash_line)
        print('Example ', i + 1)
        print(dash_line)
        print('INPUT DIALOGUE:')
        print(dataset['test'][index]['dialogue'])
        print(dash_line)
        print('BASELINE HUMAN SUMMARY:')
        print(dataset['test'][index]['summary'])
        print(dash_line)
        print()

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True,
                                  return_tensors="pt").input_ids
    return example

if __name__ == '__main__':
    huggingface_dataset_name = "knkarthick/dialogsum"
    model_name = 'google/flan-t5-base'

    print("------ view data set ----------")
    dataset_dialog = load_dataset(huggingface_dataset_name)
    print(dataset_dialog)
    # dict = {train: {}, validation: {}, test: {}}
    show_dataset_index(dataset_dialog, [40, 100])

    print("------------- load model ------------")
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load the fine-tuned model (this is download checkpoint for lab purpose)
    instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint",
                                                           torch_dtype=torch.bfloat16)

    tokenized_datasets = dataset_dialog.map(tokenize_function, batched=True)
    print(f"Shapes of the datasets:")
    print(f"Training: {tokenized_datasets['train'].shape}")
    print(f"Validation: {tokenized_datasets['validation'].shape}")
    print(f"Test: {tokenized_datasets['test'].shape}")

    print(tokenized_datasets)

    print(tokenized_datasets['validation'])
    print("---- dialog ----")
    print(tokenized_datasets['validation']['dialogue'][0])
    print("---- summary ----")
    print(tokenized_datasets['validation']['summary'][0])
    print("---- input_ids ----")
    print(tokenized_datasets['validation']['input_ids'][0])
    print("---- labels ----")
    print(tokenized_datasets['validation']['labels'][0])

    # remove the unwanted fields
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary', ])

    print("---------- create subset for lab -------")
    # to save time in lab create a subset divide by 100, 500 -> 5, 1500 -> 15 and 12500 -> 125
    # take every 100th item, with_indices=True retains the index so output index [0,100,200 ...]
    tokenized_datasets = tokenized_datasets.filter(lambda item, index: index % 100 == 0, with_indices=True)
    '''
        Shapes of the datasets:
        Training: (125, 2)
        Validation: (5, 2)
        Test: (15, 2)
        DatasetDict({
            train: Dataset({
                features: ['input_ids', 'labels'],
                num_rows: 125
            })
            validation: Dataset({
                features: ['input_ids', 'labels'],
                num_rows: 5
            })
            test: Dataset({
                features: ['input_ids', 'labels'],
                num_rows: 15
            })
        })
    '''
    print(f"Shapes of the datasets:")
    print(f"Training: {tokenized_datasets['train'].shape}")
    print(f"Validation: {tokenized_datasets['validation'].shape}")
    print(f"Test: {tokenized_datasets['test'].shape}")

    print(tokenized_datasets)

    print("---------- PEFT lora fine tune -----------")

    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )

    peft_model = get_peft_model(original_model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))
    '''
    trainable model parameters: 3538944
    all model parameters: 251116800
    percentage of trainable model parameters: 1.41%
    '''

    output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

    peft_training_args = TrainingArguments(
        output_dir=output_dir,
        auto_find_batch_size=True,
        learning_rate=1e-3,  # Higher learning rate than full fine-tuning.
        num_train_epochs=1,
        logging_steps=1,
        max_steps=1
    )

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )

    peft_trainer.train()

    peft_model_path="./peft-dialogue-summary-checkpoint-local"
    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)
    '''
    Step	Training Loss
        1	48.000000
        ('./peft-dialogue-summary-checkpoint-local/tokenizer_config.json',
         './peft-dialogue-summary-checkpoint-local/special_tokens_map.json',
         './peft-dialogue-summary-checkpoint-local/spiece.model',
         './peft-dialogue-summary-checkpoint-local/added_tokens.json',
         './peft-dialogue-summary-checkpoint-local/tokenizer.json')
    
    NOTE: check for adaptor.bin file in the local
    '''

    print("------------- use fine tuned model ----------------")
    peft_model_base = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # instead of using local using from s3 checkpoint (the local is for lab work withe subset data and quick run)
    peft_model = PeftModel.from_pretrained(peft_model_base,
                                           './peft-dialogue-summary-checkpoint-from-s3/',
                                           torch_dtype=torch.bfloat16,
                                           is_trainable=False)

    print(print_number_of_trainable_model_parameters(peft_model))
    '''
    trainable model parameters: 0
    all model parameters: 251116800
    percentage of trainable model parameters: 0.00%
    '''

