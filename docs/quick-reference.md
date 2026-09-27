# Quick Reference

[Study guide](README.md) | [Transcripts by topic](subtitles/README.md)

Use this page to refresh the course without rerunning the notebooks. For the full explanation, follow the phase link in each row.

| Topic | Remember | Revisit |
| --- | --- | --- |
| Lifecycle | Scope, select, adapt and evaluate, integrate, monitor. | [Phase 1](phase-1-foundations.md) |
| Tokens and context | Models operate on tokens; prompt, retrieved text, and examples compete for context space. | [Phase 1](phase-1-foundations.md) |
| Transformer families | Encoder-only for representations; encoder-decoder for text-to-text tasks; decoder-only for autoregressive generation. | [Phase 1](phase-1-foundations.md) |
| Pre-training | General patterns come from large datasets and substantial compute; start with a model card before choosing a model. | [Phase 1](phase-1-foundations.md) |
| Prompting | Zero-, one-, and few-shot examples steer behavior within the prompt without changing weights. | [Phase 2](phase-2-prompting.md), [Lab 1](../gen-ai/projects/summarize_dialog/Lab_1_summarize_dialogue.ipynb) |
| Decoding | `max_new_tokens`, temperature, top-k, and top-p control generation, not training. | [Phase 2](phase-2-prompting.md) |
| Evaluation | ROUGE measures reference overlap for summarization; BLEU is often used in translation. Inspect meaning and factuality as well. Benchmarks test broader skills and risks. | [Phase 2](phase-2-prompting.md) |
| Fine-tuning | Full tuning updates model weights; LoRA trains compact updates on a frozen base. Compare both to a baseline on held-out examples. | [Phase 3](phase-3-fine-tuning.md), [Lab 2](../gen-ai/projects/fine-tuning/Lab_2_fine_tune_generative_ai_model.ipynb) |
| Alignment | Preferences or a reward classifier score outputs; PPO updates the policy toward a reward, which is only a proxy for quality. | [Phase 4](phase-4-alignment.md), [Lab 3](../gen-ai/projects/rlhf/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb) |
| Optimization | Distillation changes the model; quantization changes representation precision; pruning removes weights. Measure quality after each. | [Phase 5](phase-5-applications.md) |
| Applications | RAG supplies external context; PAL uses code for calculations; ReAct-style flows coordinate reasoning and tool use. | [Phase 5](phase-5-applications.md) |

## Lab sequence in one minute

1. **Lab 1:** Hold the model fixed; improve dialogue summaries using prompts and generation settings.
2. **Lab 2:** Change the model with full fine-tuning and LoRA; compare summaries and ROUGE with the baseline.
3. **Lab 3:** Optimize a summarization-tuned model with a toxicity reward and PPO; compare before/after toxicity **and** summary usefulness.

## Pitfalls worth revisiting

- A fluent output may be false. A high overlap score may miss a negation or an invented detail.
- Larger models and lower-precision models are not automatically better for a given workload; compare on your task.
- Notebook setup cells assume the course's environment and may install older package versions. Saved outputs are examples, not fresh validation.
- Lecture examples, leaderboards, prices, and model cards can become outdated; verify current sources when building a real application.