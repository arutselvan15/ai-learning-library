# Phase 1: Foundations and Pre-training

[Study guide](README.md) | Next: [Prompting and evaluation](phase-2-prompting.md)

## In brief

An LLM learns patterns from tokens during pre-training, then can be adapted to downstream tasks. Transformers use attention to incorporate context; their architecture and training objective affect which tasks they handle naturally. Before choosing a model, consider its task fit, model card, context length, data requirements, compute, and evaluation plan.

## Learn in order

1. **Scope the problem.** Define the task and the quality, cost, and safety criteria before selecting a model. The [project lifecycle notes](../gen-ai/project-lifecycle-and-roles/README.md) show where selection, adaptation, evaluation, and integration fit.
2. **Understand tokens and pre-training.** Text is split into tokens, which may be words or parts of words. Models learn from large text collections using objectives such as masked-token prediction, next-token prediction, or span reconstruction. See the [introductory notes](../gen-ai/gen-ai.md) and [pre-training notes](../gen-ai/pretraining/README.md).
3. **Compare transformer families.** [Architecture notes](../gen-ai/model/architecture/README.md): encoder-only models (such as BERT) learn representations with masked-language modeling; encoder-decoder models (such as T5/BART) map input text to output text and suit summarization; decoder-only models (such as GPT) predict tokens autoregressively. These are tendencies, not rigid limits on what a model can do.
4. **Budget compute.** [Compute notes](../gen-ai/model/compute/README.md) discuss parameter storage, extra training memory for gradients and optimizer state, DDP versus sharding, precision formats, and the trade-off between dataset size, model size, and training compute. For FP32 weights alone, 1 billion parameters need about 4 GB (decimal); actual training memory depends on optimizer, precision, activations, and batch size. DDP replicates the model across workers; FSDP/ZeRO-style sharding distributes model state to reduce per-device memory.
5. **Select an existing model.** Check the [model-hub notes](../gen-ai/model/model-hub/README.md) and the model card for intended use, limitations, license, training data, and evaluation. The labs use FLAN-T5, an instruction-tuned encoder-decoder model, for dialogue summarization.

## Connect it to the labs

Lab 1 loads a tokenizer, the DialogSum dataset, and FLAN-T5 to generate a summary. Tokenization and the encoder-decoder architecture explain why the dialogue is input and the summary is generated output. In Lab 2, full fine-tuning updates model weights and therefore needs more training memory than inference.

## Check your understanding

- Why is an encoder-decoder model a reasonable starting point for summarization?
- What does tokenization change about input length and model cost?
- Why does a model that fits in memory for inference not necessarily fit for full training?
- When would you use an existing model rather than pre-train from scratch?

For the original lecture on the transition to transformers, see [subtitle (0)](subtitles/subtitle%20%280%29.txt). Older parameter counts and cost estimates in the notes are examples, not sizing rules for current models.