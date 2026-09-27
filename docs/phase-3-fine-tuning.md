# Phase 3: Fine-tuning and Measuring Improvement

[Study guide](README.md) | Previous: [Prompting](phase-2-prompting.md) | Next: [Alignment](phase-4-alignment.md)

## In brief

Pre-training gives a model general capabilities; instruction tuning teaches it to respond to tasks; task-specific fine-tuning adjusts its behavior using labeled examples. Unlike in-context prompting, training updates parameters. Measure improvement against a baseline and watch for degraded behavior on other tasks (catastrophic forgetting).

## Learn in order

1. Read the [fine-tuning notes](../gen-ai/fine-tuning/README.md): task-specific training, instruction tuning, and catastrophic forgetting. A single-task model may improve on summarization while losing some generality.
2. Compare **full fine-tuning** with **PEFT**. Full fine-tuning changes all trainable model weights and typically needs more memory and storage. PEFT freezes the base model and trains a small set of additional parameters. LoRA is a PEFT method that learns low-rank updates; the saved adapter must be used with a compatible base model at inference time.
3. Revisit [ROUGE and its limitations](../gen-ai/evaluation-metrics/README.md). Compute metrics on held-out data, and inspect the summaries themselves for missing facts, invented details, and changes in meaning. Do not compare numbers from different test sets as though they were the same experiment.

## Lab 2: Fine-tune a summarizer

Work through [Lab 2 notebook](../gen-ai/projects/fine-tuning/Lab_2_fine_tune_generative_ai_model.ipynb) in its own environment. The [Lab 2 guide](../gen-ai/projects/fine-tuning/README.md) links the companion scripts; the [lecture walkthrough](subtitles/subtitle%20%2821%29.txt) explains the progression from Lab 1.

1. Prepare the kernel, dependencies, DialogSum dataset, and FLAN-T5 model. The notebook includes an instance-type check for its original course environment; follow its setup cells and adapt that check only if working outside that environment.
2. Run zero-shot inference as the baseline before training.
3. Preprocess dialogue/summary pairs, perform or load full fine-tuning, and compare generated summaries by eye and with ROUGE.
4. Configure a LoRA adapter, train or load it, and repeat the qualitative and quantitative evaluation on comparable examples.
5. Compare trainable parameter counts, storage needs, and summary quality. Check whether any claimed improvement is meaningful for your use case rather than choosing the largest ROUGE value alone.

The notebook may download datasets, models, and checkpoints and may need substantial compute. Its install cells and pinned library versions are the source of truth for the original exercise; the repository's [general setup](../setup.md) does not provision its dependencies.

## What to take forward

Explain which model you would deploy for dialogue summarization, and why: baseline, fully fine-tuned model, or base model plus LoRA adapter. Save a couple of representative outputs and the conditions under which you compared them. Phase 4 builds on the idea of a summarization-tuned model but introduces a separate reward signal.

## Check your understanding

- Which parameters change in full fine-tuning versus LoRA?
- What is catastrophic forgetting, and why might PEFT reduce the risk?
- Why should human inspection accompany ROUGE when evaluating summaries?