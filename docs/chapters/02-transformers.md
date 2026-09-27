# 02. Tokens, Transformers, and Model Families

[Previous: 01. Lifecycle](01-project-lifecycle.md) | [Contents](../README.md) | Next: [03. Pre-training](03-pretraining-compute.md)

## Why transformers?

Language depends on context. The meaning of *bank*, for instance, depends on neighboring words. Older recurrent models process sequences step by step; transformers use attention to relate token representations across a sequence and are well suited to parallel training. Attention does not by itself guarantee understanding or factual accuracy.

![Encoder-only, encoder-decoder and decoder-only transformer families](../assets/model/architecture/transformers.png)

Text is split into **tokens** (often words or word pieces), converted to IDs by a tokenizer, and represented numerically by the model. Token counts depend on the tokenizer, so a word count is only a rough estimate of context and usage cost. An embedding is a learned vector representation; token IDs themselves are not embeddings.

## Choose the architecture by objective

| Family | Pre-training objective | Typical strength | Examples |
| --- | --- | --- | --- |
| Encoder-only | Recover masked tokens using surrounding context | Text representations and classification | BERT, RoBERTa |
| Encoder-decoder (sequence-to-sequence) | Encode input and generate target text; T5 uses span corruption | Summarization and translation | T5, FLAN-T5, BART |
| Decoder-only | Predict the next token using preceding context | Open-ended generation | GPT-style models, BLOOM |

![Masked-token prediction with bidirectional context](../assets/model/architecture/encoder.png)

![T5-style span reconstruction with an encoder and decoder](../assets/model/architecture/encoder-decoder.png)

![Next-token prediction using previous tokens](../assets/model/architecture/decoder.png)

FLAN-T5 is an instruction-tuned T5 model. The lab feeds it a dialogue as input and generates a shorter text as output. Architecture names describe common training setups, not strict limits on downstream uses. Even a model that supports long context can still omit or distort a fact.

## Context versus training

At inference time, the model consumes a prompt within its supported context length and generates tokens. In-context examples steer a response **without** changing weights. Pre-training and fine-tuning **do** change weights. Keeping this distinction clear makes the next chapters much easier.

**Checkpoint:** Explain why an encoder-decoder model suits dialogue summarization and why the tokenizer matters when a dialogue is long.

Sources: [transformer motivation](../subtitles/subtitle%20%280%29.txt), [attention](../subtitles/subtitle%20%281%29.txt), [architectures and tokenization](../subtitles/subtitle%20%282%29.txt). See the architecture illustrations above when comparing model families.