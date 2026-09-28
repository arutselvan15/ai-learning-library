import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import truststore

# Use macOS system keychain (includes corporate proxy CAs like Cisco Secure Access).
# certifi alone only has public CAs and fails behind SSL inspection.
truststore.inject_into_ssl()

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def load_model(name):
    org_model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(name)

    return org_model

if __name__ == '__main__':
    model_name = 'google/flan-t5-base'
    original_model = load_model(model_name)
    print(print_number_of_trainable_model_parameters(original_model))

'''
output:
trainable model parameters: 247577856
all model parameters: 247577856
percentage of trainable model parameters: 100.00%
'''


