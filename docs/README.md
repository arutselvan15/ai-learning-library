# Generative AI with Large Language Models

A chapter-by-chapter knowledge base built from the course transcripts, personal notes, three labs, and slide illustrations. Start with Chapter 01 and continue in order; use the [quick reference](quick-reference.md) to refresh a topic later. Chapters contain the explanation, relevant images, original transcript links, and lab instructions together.

## Phase 1: Understand and select an LLM

1. [01. The project lifecycle](chapters/01-project-lifecycle.md): define a task, success criteria, and the path to deployment.
2. [02. Tokens and transformers](chapters/02-transformers.md): attention, tokenization, and encoder/decoder families.
3. [03. Pre-training and compute](chapters/03-pretraining-compute.md): model cards, memory, distributed training, and scaling.

## Phase 2: Prompt and evaluate

4. [04. Prompting and generation](chapters/04-prompting.md): instructions, zero-/one-/few-shot examples, and decoding settings.
5. [05. Lab 1: dialogue summarization](chapters/05-lab-prompting.md): compare prompted FLAN-T5 summaries without training.
6. [06. Evaluation](chapters/06-evaluation.md): ROUGE, BLEU, human review, and holistic benchmarks.

## Phase 3: Adapt a model

7. [07. Fine-tuning and PEFT](chapters/07-fine-tuning.md): instruction tuning, full tuning, LoRA, and forgetting.
8. [08. Lab 2: fine-tune a summarizer](chapters/08-lab-fine-tuning.md): compare full tuning and LoRA against a baseline.

## Phase 4: Align behavior

9. [09. RLHF](chapters/09-rlhf.md): preferences, rewards, PPO, and limitations.
10. [10. Lab 3: less-toxic summaries](chapters/10-lab-rlhf.md): evaluate a classifier reward and summary quality.

## Phase 5: Build an application

11. [11. Serving optimization](chapters/11-optimization.md): distillation, quantization, and pruning.
12. [12. LLM applications](chapters/12-applications.md): RAG, program-aided computation, tools, and orchestration.

## Sources and use

- [Transcript index](lecture-transcripts/README.md) organizes all 43 original lecture transcript files; these are preserved as raw primary sources. Chapters link the most relevant lectures directly.
- [Week 1](slides/Weel1.pdf), [Week 2](slides/Week2.pdf), and [Week 3](slides/Week3.pdf) are the original PDFs (including the source filename `Weel1.pdf`). Chapters embed selected slide-derived images from `assets/` and link the appropriate PDF.
- [Lab 1](projects/summarize_dialog/Lab_1_summarize_dialogue.ipynb), [Lab 2](projects/fine-tuning/Lab_2_fine_tune_generative_ai_model.ipynb), and [Lab 3](projects/rlhf/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb) are the original exercises. Each lab chapter explains what to do and how to compare results.
- [Setup for standalone examples](setup.md) is separate from notebook setup; use the dependency and hardware guidance in each notebook to run the labs. No notebook is automatically executed by reading these docs.
- [Further resources and repository examples](resources.md) collect optional provider, model-hub, and SDK links from the old notes.

The course uses historical examples. Check current model cards, library documentation, and evaluation datasets before adopting its model sizes, prices, or benchmark scores for a real project.
