import truststore
import torch
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer

# Use macOS system keychain (includes corporate proxy CAs like Cisco Secure Access).
# certifi alone only has public CAs and fails behind SSL inspection.
truststore.inject_into_ssl()

from datasets import load_dataset

tokenizer = None

def show_dataset_index(dataset, indexes):
    dash_line = '-'.join('' for x in range(100))

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


def create_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)

'''
format:

Training prompt (dialogue):

Summarize the following conversation.

    Chris: This is his part of the conversation.
    Antje: This is her part of the conversation.
    
Summary: 
Training response (summary):

Both Chris and Antje participated in the conversation.




'''
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
    show_dataset_index(dataset_dialog, [40,100])

    print("------------- load model ------------")
    original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("--------- tokenize the dialog ---------")
    # The dataset actually contains 3 diff splits: train, validation, test.
    # The tokenize_function code is handling all data across all splits in batches.
    # map function apply the given function for all items in the dialog dataset map
    '''
    
        Shapes of the datasets:
        Training: (12460, 6)
        Validation: (500, 6)
        Test: (1500, 6)
        DatasetDict({
            train: Dataset({
                features: ['id', 'dialogue', 'summary', 'topic', 'input_ids', 'labels'],
                num_rows: 12460
            })
            validation: Dataset({
                features: ['id', 'dialogue', 'summary', 'topic', 'input_ids', 'labels'],
                num_rows: 500
            })
            test: Dataset({
                features: ['id', 'dialogue', 'summary', 'topic', 'input_ids', 'labels'],
                num_rows: 1500
            })
        })


        Dataset({
            features: ['id', 'dialogue', 'summary', 'topic', 'input_ids', 'labels'],
            num_rows: 500
        })
        ---- dialog ----
        #Person1#: Hello, how are you doing today?
        #Person2#: I ' Ve been having trouble breathing lately.
        #Person1#: Have you had any type of cold lately?
        #Person2#: No, I haven ' t had a cold. I just have a heavy feeling in my chest when I try to breathe.
        #Person1#: Do you have any allergies that you know of?
        #Person2#: No, I don ' t have any allergies that I know of.
        #Person1#: Does this happen all the time or mostly when you are active?
        #Person2#: It happens a lot when I work out.
        #Person1#: I am going to send you to a pulmonary specialist who can run tests on you for asthma.
        #Person2#: Thank you for your help, doctor.
        ---- summary ----
        #Person2# has trouble breathing. The doctor asks #Person2# about it and will send #Person2# to a pulmonary specialist.
        ---- input_ids ----
        [12198, 1635, 1737, 8, 826, 3634, 5, 1713, 345, 13515, 536, 4663, 10, 8774, 6, 149, 33, 25, 692, 469, 58, 1713, 345, 13515, 357, 4663, 10, 27, 3, 31, 3901, 118, 578, 3169, 10882, 12643, 5, 1713, 345, 13515, 536, 4663, 10, 2114, 25, 141, 136, 686, 13, 2107, 12643, 58, 1713, 345, 13515, 357, 4663, 10, 465, 6, 27, 43, 29, 3, 31, 3, 17, 141, 3, 9, 2107, 5, 27, 131, 43, 3, 9, 2437, 1829, 16, 82, 5738, 116, 27, 653, 12, 13418, 5, 1713, 345, 13515, 536, 4663, 10, 531, 25, 43, 136, 18500, 24, 25, 214, 13, 58, 1713, 345, 13515, 357, 4663, 10, 465, 6, 27, 278, 3, 31, 3, 17, 43, 136, 18500, 24, 27, 214, 13, 5, 1713, 345, 13515, 536, 4663, 10, 3520, 48, 1837, 66, 8, 97, 42, 3323, 116, 25, 33, 1676, 58, 1713, 345, 13515, 357, 4663, 10, 94, 2906, 3, 9, 418, 116, 27, 161, 91, 5, 1713, 345, 13515, 536, 4663, 10, 27, 183, 352, 12, 1299, 25, 12, 3, 9, 3, 26836, 4253, 113, 54, 661, 3830, 30, 25, 21, 17688, 5, 1713, 345, 13515, 357, 4663, 10, 1562, 25, 21, 39, 199, 6, 2472, 5, 20698, 10, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ---- labels ----
        [1713, 345, 13515, 357, 4663, 65, 3169, 10882, 5, 37, 2472, 987, 7, 1713, 345, 13515, 357, 4663, 81, 34, 11, 56, 1299, 1713, 345, 13515, 357, 4663, 12, 3, 9, 3, 26836, 4253, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    '''
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

    print ("---------- create subset for lab -------")
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

    print("---------- prepare for fine tuning ----------")
    output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=1,
        max_steps=1
    )

    trainer = Trainer(
        model=original_model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation']
    )

    print("------ start the training --------")
    trainer.train()
    '''
        Step	Training Loss
            1	47.250000

        TrainOutput(global_step=1, training_loss=47.25, metrics={'train_runtime': 73.4695, 'train_samples_per_second': 0.109, 'train_steps_per_second': 0.014, 'total_flos': 5478058819584.0, 'train_loss': 47.25, 'epoch': 0.06})
    '''