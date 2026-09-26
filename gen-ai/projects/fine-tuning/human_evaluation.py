import truststore
import torch
import time
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

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



if __name__ == '__main__':
    huggingface_dataset_name = "knkarthick/dialogsum"
    model_name = 'google/flan-t5-base'

    print("------ view data set ----------")
    dataset_dialog = load_dataset(huggingface_dataset_name)
    print(dataset_dialog)
    # dict = {train: {}, validation: {}, test: {}}
    show_dataset_index(dataset_dialog, [40, 100])

    print("------------- load models ------------")
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load the fine-tuned model (this is download checkpoint for lab purpose)
    instruct_model = AutoModelForSeq2SeqLM.from_pretrained("./flan-dialogue-summary-checkpoint",
                                                           torch_dtype=torch.bfloat16)
    # instead of using local using from s3 checkpoint (the local is for lab work withe subset data and quick run)
    peft_model = PeftModel.from_pretrained(original_model,
                                           './peft-dialogue-summary-checkpoint-from-s3/',
                                           torch_dtype=torch.bfloat16,
                                           is_trainable=False)


    print("----------- test 1 -------------")
    index = 200
    dialogue = dataset_dialog['test'][index]['dialogue']
    human_baseline_summary = dataset_dialog['test'][index]['summary']

    prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary:
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    print("----- process original --------")
    original_model_outputs = original_model.generate(input_ids=input_ids,
                                                     generation_config=GenerationConfig(max_new_tokens=200,
                                                                                        num_beams=1))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

    print("---------- process full fine tune --------")
    instruct_model_outputs = instruct_model.generate(input_ids=input_ids,
                                                     generation_config=GenerationConfig(max_new_tokens=200,
                                                                                        num_beams=1))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

    print("--------- process perf lora fine tune ---------")
    peft_model_outputs = peft_model.generate(input_ids=input_ids,
                                             generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
    print(dash_line)
    print(f'ORIGINAL MODEL:\n{original_model_text_output}')
    print(dash_line)
    print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')
    print(dash_line)
    print(f'PEFT MODEL: {peft_model_text_output}')
    '''
    ---------------------------------------------------------------------------------------------------
    BASELINE HUMAN SUMMARY:
    #Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
    ---------------------------------------------------------------------------------------------------
    ORIGINAL MODEL:
    #Person1: Have you considered upgrading your computer? #Person2: I'm not sure what exactly I would need. #Person1: I'd like to make my own flyers and banners. #Person2: That would be a definite bonus. #Person1: You could consider adding a painting program to your software. #Person2: I'm not sure what I would need. #Person1: I'd like to make my own flyers and banners. #Person2: I'd like to make my own banners. #Person1: I'm not sure what I need. #Person2: I'd like to make my own flyers and banners. #Person1: I'd like to make my own banners and banners. #Person1: I'd like to make my own flyers. #Person1
    ---------------------------------------------------------------------------------------------------
    INSTRUCT MODEL:
    #Person1# suggests #Person2# adding a painting program to #Person2#'s software and upgrading the hardware. #Person2# also wants to add a CD-ROM drive.
    ---------------------------------------------------------------------------------------------------
    PEFT MODEL: #Person1# recommends adding a painting program to #Person2#'s software and upgrading hardware. #Person2# also wants to upgrade the hardware because it's outdated now.
    '''