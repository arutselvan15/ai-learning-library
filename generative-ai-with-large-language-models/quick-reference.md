# Quick Reference

[Chapters](README.md) | [Course resources](resources.md)

| Question | Short answer | Chapter |
| --- | --- | --- |
| Where do I start? | Define the task and evaluation criteria before choosing a model. | [Chapter 1: Lifecycle](phases/01-understand-and-select/01-project-lifecycle/README.md) |
| Which model family? | Encoder-only for representations, encoder-decoder for input-to-output text, decoder-only for next-token generation. | [Chapter 2: Transformers](phases/01-understand-and-select/02-transformers/README.md) |
| What makes a model expensive to train? | Weights plus gradients, optimizer state, activations, data, and compute. | [Chapter 3: Pre-training](phases/01-understand-and-select/03-pretraining-compute/README.md) |
| Prompt or train? | Prompt first; in-context examples do not update weights. | [Chapter 1: Prompting](phases/02-prompt-and-evaluate/01-prompting/README.md), [Chapter 2: Lab 1](phases/02-prompt-and-evaluate/02-lab-prompting/README.md) |
| How do I judge a summary? | Use held-out examples, human review, and task metrics; ROUGE alone misses changed meaning. | [Chapter 3: Evaluation](phases/02-prompt-and-evaluate/03-evaluation/README.md) |
| Full tuning or LoRA? | Full tuning updates the model; LoRA trains smaller task-specific updates to a frozen base. Compare quality and resource cost. | [Chapter 1: Fine-tuning](phases/03-adapt-a-model/01-fine-tuning/README.md), [Chapter 2: Lab 2](phases/03-adapt-a-model/02-lab-fine-tuning/README.md) |
| What does RLHF optimize? | A reward signal based on preferences or a proxy; the score is not the same as overall quality. | [Chapter 1: RLHF](phases/04-align-behavior/01-rlhf/README.md), [Chapter 2: Lab 3](phases/04-align-behavior/02-lab-rlhf/README.md) |
| How do I reduce serving cost? | Measure distillation, quantization, or pruning on the target workload and hardware. | [Chapter 1: Optimization](phases/05-build-an-application/01-optimization/README.md) |
| How do I use current facts or exact arithmetic? | Retrieve authorized sources for facts; call a validated computation tool for arithmetic. | [Chapter 2: Applications](phases/05-build-an-application/02-applications/README.md) |

**Lab order:** [Prompt only](phases/02-prompt-and-evaluate/02-lab-prompting/README.md) -> [full tuning and LoRA](phases/03-adapt-a-model/02-lab-fine-tuning/README.md) -> [reward-based PPO](phases/04-align-behavior/02-lab-rlhf/README.md). Keep representative input/output examples and the evaluation conditions for each comparison.

**Common trap:** Fluency is not correctness; a high n-gram overlap can hide a negation, and a lower toxicity score alone cannot prove a summary is useful or safe.