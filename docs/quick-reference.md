# Quick Reference

[Chapters](README.md) | [Transcript index](subtitles/README.md)

| Question | Short answer | Chapter |
| --- | --- | --- |
| Where do I start? | Define the task and evaluation criteria before choosing a model. | [01. Lifecycle](chapters/01-project-lifecycle.md) |
| Which model family? | Encoder-only for representations, encoder-decoder for input-to-output text, decoder-only for next-token generation. | [02. Transformers](chapters/02-transformers.md) |
| What makes a model expensive to train? | Weights plus gradients, optimizer state, activations, data, and compute. | [03. Pre-training](chapters/03-pretraining-compute.md) |
| Prompt or train? | Prompt first; in-context examples do not update weights. | [04. Prompting](chapters/04-prompting.md), [05. Lab 1](chapters/05-lab-prompting.md) |
| How do I judge a summary? | Use held-out examples, human review, and task metrics; ROUGE alone misses changed meaning. | [06. Evaluation](chapters/06-evaluation.md) |
| Full tuning or LoRA? | Full tuning updates the model; LoRA trains smaller task-specific updates to a frozen base. Compare quality and resource cost. | [07. Fine-tuning](chapters/07-fine-tuning.md), [08. Lab 2](chapters/08-lab-fine-tuning.md) |
| What does RLHF optimize? | A reward signal based on preferences or a proxy; the score is not the same as overall quality. | [09. RLHF](chapters/09-rlhf.md), [10. Lab 3](chapters/10-lab-rlhf.md) |
| How do I reduce serving cost? | Measure distillation, quantization, or pruning on the target workload and hardware. | [11. Optimization](chapters/11-optimization.md) |
| How do I use current facts or exact arithmetic? | Retrieve authorized sources for facts; call a validated computation tool for arithmetic. | [12. Applications](chapters/12-applications.md) |

**Lab order:** [Prompt only](chapters/05-lab-prompting.md) -> [full tuning and LoRA](chapters/08-lab-fine-tuning.md) -> [reward-based PPO](chapters/10-lab-rlhf.md). Keep representative input/output examples and the evaluation conditions for each comparison.

**Common trap:** Fluency is not correctness; a high n-gram overlap can hide a negation, and a lower toxicity score alone cannot prove a summary is useful or safe.