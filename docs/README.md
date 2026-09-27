# Generative AI with Large Language Models: Study Guide

This guide turns the course notes and three notebooks into a learning path. Read the phases in order on a first pass; use the phase summaries and the [quick reference](quick-reference.md) for revision. The notebooks are the hands-on exercises; the Python scripts alongside them are smaller explorations of the same ideas.

## Course at a glance

1. **Understand the model:** transformers, model families, pre-training, compute, and model selection.
2. **Use and adapt the model:** prompting, generation settings, evaluation, and fine-tuning.
3. **Align and deploy the model:** human feedback, RLHF, optimization, and application patterns.

These stages follow the course's project lifecycle: scope a use case, select a model, adapt and evaluate it, then integrate and optimize it. See the [project lifecycle notes](../gen-ai/project-lifecycle-and-roles/README.md).

## First-time learning path

| Order | Phase | Learn | Practice |
| --- | --- | --- | --- |
| 1 | [Foundations and pre-training](phase-1-foundations.md) | Transformers, model architectures, tokenization, pre-training, compute, and model hubs | Identify an appropriate model and its resource constraints |
| 2 | [Prompting and evaluation](phase-2-prompting.md) | Zero-, one-, and few-shot prompts, generation controls, ROUGE, BLEU, and benchmarks | [Lab 1: summarize dialogue](../gen-ai/projects/summarize_dialog/Lab_1_summarize_dialogue.ipynb) |
| 3 | [Fine-tuning](phase-3-fine-tuning.md) | Instruction tuning, full fine-tuning, PEFT/LoRA, catastrophic forgetting, and evaluation | [Lab 2: fine-tune a summarizer](../gen-ai/projects/fine-tuning/Lab_2_fine_tune_generative_ai_model.ipynb) |
| 4 | [Alignment with human feedback](phase-4-alignment.md) | Preferences, reward models, RLHF, PPO, and limitations | [Lab 3: detoxify summaries](../gen-ai/projects/rlhf/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb) |
| 5 | [Optimization and applications](phase-5-applications.md) | Quantization, pruning, distillation, RAG, tools, and orchestration | Sketch an end-to-end LLM application |

For each phase, read its summary and linked notes, work through the notebook where listed, then answer its review questions before continuing. The labs build conceptually on the same dialogue summarization task; follow their own setup and model-loading cells rather than assuming one notebook's kernel state carries into the next.

## Sources and scope

- [Topic notes](../gen-ai/README.md) contain the detailed course material and personal notes; phase pages provide the reading order and connections.
- [Lab guides](../gen-ai/projects/README.md) link the notebooks and companion scripts. [Local setup](../setup.md) covers the repository's lightweight Python examples, not necessarily the full notebook environments.
- [Course transcripts by phase](subtitles/README.md) index the raw, numbered subtitle files by topic. The attached slide images illustrate concepts, but the slides folder has no files to link at present.

The course examples and model sizes are historical snapshots. Check current model cards, library docs, and benchmark leaderboards before making deployment decisions.
