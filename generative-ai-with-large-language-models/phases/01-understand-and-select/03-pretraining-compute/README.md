# Chapter 3 - Pre-training, Model Selection, and Compute

[Previous: Chapter 2: Transformers](../02-transformers/README.md) | [Contents](../../../README.md) | Next: [Phase 2, Chapter 1: Prompting](../../02-prompt-and-evaluate/01-prompting/README.md)

## From data to a foundation model

Pre-training learns general patterns from a large, quality-filtered text corpus. A tokenizer maps text to IDs; the chosen architecture's learning objective teaches the model to predict missing spans or future tokens. Training a general model from scratch requires substantial data, time, and hardware. A specialized domain with unfamiliar terminology may justify additional domain training, but adapting an existing model is usually the first experiment.

The objective depends on the architecture. **Masked language modeling** hides selected tokens and trains an encoder to reconstruct them from both sides. **Causal language modeling** predicts the next token using only preceding tokens, repeating the process across the sequence. **Span corruption**, used by T5, replaces a contiguous span with a sentinel token and trains the encoder-decoder model to reconstruct the missing span. These objectives explain why encoder-only, decoder-only, and sequence-to-sequence models behave differently.

![Pre-training data, quality filtering, tokens, model and vocabulary](assets/llm-pre-training.png)

Choose a model by **task fit**, not only parameter count. A model card should state intended uses, limitations, training/evaluation information, license, and any access restrictions. Test the model on your own representative examples. The course labs choose instruction-tuned FLAN-T5 for text-to-text summarization. [Model selection lecture](transcripts/model-selection.md) and [domain pre-training lecture](transcripts/domain-specific-pretraining.md) explain the trade-off.

## Memory and throughput

At FP32, one billion weights occupy roughly $10^9 \times 4$ bytes, or 4 GB in decimal units, just to store the weights. Training also needs gradients, optimizer states, and activations, whose sizes depend on the optimizer, precision, batch, and implementation. The often quoted ~24 GB for a one-billion-parameter training example is an illustrative estimate, **not** a guarantee that a model will fit.

![Illustrative breakdown of GPU memory for weights, optimizer, gradients and activations](assets/1b-2.png)

Lower numerical precision can save memory: FP32 stores 32-bit floating-point values, FP16 and BF16 use 16 bits, while INT8 uses 8-bit integers plus scaling information in practical quantization schemes. BF16 has a wider exponent range than FP16 but less mantissa precision. Quantization can affect output quality and does not make every operation or training setup use one quarter the memory.

BF16 keeps the eight-bit exponent range of FP32 but uses fewer fraction bits, so it can represent a similarly wide range of magnitudes with less precision. FP16 has more fraction bits than BF16 but a smaller exponent range and can overflow more easily for some training values. The choice is therefore a numerical-stability and hardware decision, not just a bit-count decision.

![Numerical formats and their approximate value ranges](assets/quantization.png)

When using multiple GPUs, **DDP** replicates the model on each worker and synchronizes gradients after each worker sees different data. **FSDP** and ZeRO-style sharding split parameters, gradients, and/or optimizer states across workers, reducing memory per GPU at the cost of communication and complexity. More GPUs do not automatically make training efficient.

![Distributed data parallel training and gradient synchronization](assets/ddp.png)

## Compute-optimal training

With a fixed compute budget, spending everything on model size can leave too few training tokens. The Chinchilla result argued for balancing parameter count and training tokens (roughly 20 tokens per parameter in its studied regime). It is a historical research finding, not a universal deployment formula.

The practical lesson is to ask whether a large model is under-trained before assuming that adding parameters is the best use of compute. Data availability, quality, deduplication, and the target domain can prevent the theoretical ratio from being reached. Domain pre-training is most defensible when important terminology or writing conventions are poorly represented in general data, as in legal, medical, or financial language; it also requires enough trustworthy domain data to justify the cost.

![Historical comparison of parameter and token counts](assets/chinchilla-2.png)

**Checkpoint:** Why can a model fit for inference but fail to fit for full fine-tuning? When would DDP replication be insufficient? Which model-card details would you check before using FLAN-T5?

Sources: [GPU memory and precision](transcripts/gpu-memory-and-precision.md), [scaling and compute](transcripts/compute-budgets-and-scaling.md); [Week 1 slides](../../../slides/Weel1.pdf). Model counts, GPU estimates, and leaderboards in course materials are historical.