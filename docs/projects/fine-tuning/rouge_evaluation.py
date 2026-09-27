import truststore
import torch
import evaluate
import pandas as pd
import numpy as np
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

    print("------------- load model ------------")
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
    ## run random 10 dialog
    print("----------- run dialogs -------------")
    rouge = evaluate.load('rouge')

    dialogues = dataset_dialog['test'][0:10]['dialogue']
    human_baseline_summaries = dataset_dialog['test'][0:10]['summary']

    original_model_summaries = []
    instruct_model_summaries = []
    peft_model_summaries = []

    for _, dialogue in enumerate(dialogues):
        prompt = f"""
    Summarize the following conversation.

    {dialogue}

    Summary: """
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        original_model_outputs = original_model.generate(input_ids=input_ids,
                                                         generation_config=GenerationConfig(max_new_tokens=200))
        original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

        instruct_model_outputs = instruct_model.generate(input_ids=input_ids,
                                                         generation_config=GenerationConfig(max_new_tokens=200))
        instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)

        peft_model_outputs = peft_model.generate(input_ids=input_ids,
                                                 generation_config=GenerationConfig(max_new_tokens=200))
        peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

        original_model_summaries.append(original_model_text_output)
        instruct_model_summaries.append(instruct_model_text_output)
        peft_model_summaries.append(peft_model_text_output)

    zipped_summaries = list(
        zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries, peft_model_summaries))

    df = pd.DataFrame(zipped_summaries,
                      columns=['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries',
                               'peft_model_summaries'])
    print(df)


    print("-------- evaluate -------------")
    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    instruct_model_results = rouge.compute(
        predictions=instruct_model_summaries,
        references=human_baseline_summaries[0:len(instruct_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries[0:len(peft_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    print(original_model_results)
    print('INSTRUCT MODEL:')
    print(instruct_model_results)
    print('PEFT MODEL:')
    print(peft_model_results)

    '''
ORIGINAL MODEL:
{'rouge1': 0.20065891231275845, 'rouge2': 0.07542028985507246, 'rougeL': 0.17947738728507956, 'rougeLsum': 0.18211293976678591}
INSTRUCT MODEL:
{'rouge1': 0.4015906463624618, 'rouge2': 0.17568542724181807, 'rougeL': 0.2874569966059625, 'rougeLsum': 0.2886327613084294}
PEFT MODEL:
{'rouge1': 0.3725351062275605, 'rouge2': 0.12138811933618107, 'rougeL': 0.27620639623170606, 'rougeLsum': 0.2758134870822362}
    '''

    print("---------- evaluate all results (stored in local file ---------")
    results = pd.read_csv("data/dialogue-summary-training-results.csv")

    human_baseline_summaries = results['human_baseline_summaries'].values
    original_model_summaries = results['original_model_summaries'].values
    instruct_model_summaries = results['instruct_model_summaries'].values
    peft_model_summaries = results['peft_model_summaries'].values

    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    instruct_model_results = rouge.compute(
        predictions=instruct_model_summaries,
        references=human_baseline_summaries[0:len(instruct_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=human_baseline_summaries[0:len(peft_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    print('ORIGINAL MODEL:')
    print(original_model_results)
    print('INSTRUCT MODEL:')
    print(instruct_model_results)
    print('PEFT MODEL:')
    print(peft_model_results)
    '''
ORIGINAL MODEL:
{'rouge1': 0.2334158581572823, 'rouge2': 0.07603964187010573, 'rougeL': 0.20145520923859048, 'rougeLsum': 0.20145899339006135}
INSTRUCT MODEL:
{'rouge1': 0.42161291557556113, 'rouge2': 0.18035380596301792, 'rougeL': 0.3384439349963909, 'rougeLsum': 0.33835653595561666}
PEFT MODEL:
{'rouge1': 0.40810631575616746, 'rouge2': 0.1633255794568712, 'rougeL': 0.32507074586565354, 'rougeLsum': 0.3248950182867091}
    '''

    print("Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")

    improvement = (np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values())))
    for key, value in zip(peft_model_results.keys(), improvement):
        print(f'{key}: {value * 100:.2f}%')

    '''
    Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL
    rouge1: 17.47%
    rouge2: 8.73%
    rougeL: 12.36%
    rougeLsum: 12.34%
    '''

    print("Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL")

    improvement = (np.array(list(peft_model_results.values())) - np.array(list(instruct_model_results.values())))
    for key, value in zip(peft_model_results.keys(), improvement):
        print(f'{key}: {value * 100:.2f}%')

        '''
        Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL
rouge1: -1.35%
rouge2: -1.70%
rougeL: -1.34%
rougeLsum: -1.35%

Here you see a small percentage decrease in the ROUGE metrics vs. full fine-tuned. However, the training requires much less computing and memory resources (often just a single GPU).
        '''