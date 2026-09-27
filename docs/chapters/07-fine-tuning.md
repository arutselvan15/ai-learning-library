# 07. Instruction and Parameter-efficient Fine-tuning

[Previous: 06. Evaluation](06-evaluation.md) | [Contents](../README.md) | Next: [08. Lab 2](08-lab-fine-tuning.md)

## When prompting is not enough

Pre-training produces a general model. Instruction tuning exposes it to task instructions paired with desired responses; task-specific fine-tuning provides examples for a particular goal. Prompting supplies examples at inference time without changing weights. Fine-tuning **updates** weights, so it needs training data and validation on unseen inputs.

![Single-task fine-tuning turns a pretrained model into a task-adapted one](../assets/fine-tuning/single-task.png)

For dialogue summarization, training pairs are dialogues and reference summaries. If a specific summary style or required fields cannot be obtained consistently by prompting, fine-tuning may help. First verify that failures are not caused by missing input context or a bad evaluation criterion.

## Full fine-tuning versus PEFT

| Method | Trained parameters | Strength | Trade-off |
| --- | --- | --- | --- |
| Full fine-tuning | All model weights | Maximum adaptation flexibility | More compute and memory; full model checkpoint per task |
| LoRA (a PEFT method) | Small low-rank updates; base model remains frozen | Smaller trainable state and task adapter | Adapter depends on the base model; may not match full tuning on every task |

![Full fine-tuning stores a model per task](../assets/fine-tuning/full-ft.png)

![PEFT stores small task-specific weights with a shared base model](../assets/fine-tuning/peft-2.png)

Full training on only one task can cause **catastrophic forgetting**, degrading performance on unrelated tasks. Multitask instruction tuning mixes examples from several tasks to help retain wider capabilities, but requires suitable data and evaluation across tasks. PEFT can reduce the scope of model changes; it does not guarantee immunity from forgetting or eliminate inference cost.

![A model trained on one task may lose ability on another](../assets/fine-tuning/catas.png)

**Checkpoint:** Which technique would you choose for three specialized summarization styles on the same base model? What would you test besides task-specific ROUGE before deploying it?

Sources: [instruction tuning introduction](../subtitles/subtitle%20%2811%29.txt), [single-task tuning](../subtitles/subtitle%20%2813%29.txt), [multitask tuning](../subtitles/subtitle%20%2814%29.txt), [PEFT](../subtitles/subtitle%20%2818%29.txt), [LoRA](../subtitles/subtitle%20%2819%29.txt), [prompt tuning](../subtitles/subtitle%20%2820%29.txt); [Week 2 slides](../slides/Week2.pdf).