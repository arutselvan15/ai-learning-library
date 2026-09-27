# 05. Lab 1: Summarize Dialogue with Prompts

[Previous: 04. Prompting](04-prompting.md) | [Contents](../README.md) | Next: [06. Evaluation](06-evaluation.md)

## Goal and materials

Compare FLAN-T5 summaries of DialogSum conversations with a bare prompt, explicit instructions, one or several examples, and different generation settings. The model weights stay fixed. Work through [the original Lab 1 notebook](../projects/summarize_dialog/Lab_1_summarize_dialogue.ipynb); the scripts in its folder are optional, smaller experiments. Read the [lab walkthrough transcript](../subtitles/subtitle%20%286%29.txt) when a notebook step needs more context.

## Follow the notebook in this order

1. **Environment:** Choose the kernel and install packages as directed in the notebook. Its instance check expects the original AWS `ml.m5.2xlarge` environment (8 vCPUs, roughly 32 GiB RAM); local environments may need different dependencies, hardware, or an adapted check. [Repository setup](../setup.md) covers separate API examples, not these labs.
2. **Data and model:** Load [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum), the FLAN-T5 model and its tokenizer. Look at dialogue/summary pairs before generating anything. Token IDs come from the tokenizer; encoding a string is not itself the embedding layer.
3. **Baseline:** Generate a summary without a task-specific instruction. Record the input, output, and what is missing or incorrect.
4. **Zero-shot instruction:** Try an explicit instruction and then the FLAN-T5 template. Keep the same example dialogue for comparison.
5. **One-shot and few-shot:** Add one or more labeled examples. Observe both summary quality and the growing prompt length.
6. **Generation configuration:** Vary decoding settings one at a time and note whether a change improves fidelity or only surface style.

Optional focused scripts: [dataset loading](../projects/summarize_dialog/load_dataset.py), [tokenizer](../projects/summarize_dialog/tokenizer.py), [zero-shot baseline](../projects/summarize_dialog/zero_shot_inference_1.py), [one-shot](../projects/summarize_dialog/one_shot_inference.py), [few-shot](../projects/summarize_dialog/few_shot_inference.py), [generation settings](../projects/summarize_dialog/generative_config_params.py). The notebook is the primary exercise; scripts may require independent setup.

## Capture your result

Create a comparison for the **same dialogue**: prompt method, input-token usage (if available), generated summary, and factual errors. A successful result is shorter than the dialogue while retaining relevant people, decisions, and actions. Do not infer general model quality from one example. The next chapter explains evaluation beyond a visual comparison.

**Checkpoint:** Which change improved the summary most: clearer instruction, demonstrations, or decoding? Could a score based only on word overlap detect a reversed decision?