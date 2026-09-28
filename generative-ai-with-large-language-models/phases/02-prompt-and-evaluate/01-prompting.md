# Chapter 1 - Prompting and Generation

[Previous: Phase 1, Chapter 3: Pre-training](../01-understand-and-select/03-pretraining-compute.md) | [Contents](../../README.md) | Next: [Chapter 2: Lab 1](02-lab-prompting.md)

## First try the model as it is

A **prompt** is input to the model; **inference** is the act of producing a response; a **completion** is generated output. Prompt engineering changes the instruction, context, or examples rather than the trained weights. An explicit instruction such as "Summarize this dialogue" is usually a better baseline than passing a dialogue alone.

| Technique | What the prompt contains | Main trade-off |
| --- | --- | --- |
| Zero-shot | Instruction and task input, no worked examples | Cheapest context usage; may miss a task-specific format |
| One-shot | One example input and desired output | Demonstrates format but consumes context |
| Few-shot | Several demonstrations | May clarify the task, but costs tokens and may crowd out the actual input |

In-context learning is not fine-tuning: if you start a fresh request without the demonstrations, the model has not learned new weights. The right number of examples depends on model, task, context budget, and evaluation; there is no fixed cutoff after five or six examples.

## Control decoding

- `max_new_tokens` caps the generated length, not the total input length.
- Temperature changes the shape of the next-token distribution; a lower value usually reduces randomness.
- Top-k samples from the highest-probability $k$ candidates; top-p samples from a set whose cumulative probability reaches a threshold $p$.
- Generation controls do not update the model or supply missing facts. Some combinations and defaults depend on the library and whether sampling is enabled.

Greedy decoding repeatedly selects the single most likely next token. It is deterministic, but it can fall into repeated words or repeated sequences. Sampling gives lower-probability candidates a chance to be selected; for example, a token with probability $0.02$ has about a 2% chance on a draw from the active distribution. In Hugging Face Transformers, temperature, top-k, and top-p sampling generally require `do_sample=True`; always check the library's defaults rather than assuming a parameter is active.

Keep the same dialogue and evaluation criteria when comparing prompts. For a summary, check who said what, important actions, omissions, and invented details. A natural-sounding but unfaithful summary is not a success.

**Checkpoint:** Write a zero-shot summarization instruction, then add one example. What extra tokens does the demonstration cost, and what behavior do you expect it to clarify?

Sources: [in-context prompting lecture](../../lecture-transcripts/zero-shot-and-in-context-prompting.md), [generation controls lecture](../../lecture-transcripts/generation-settings.md); [Week 1 slides](../../slides/Weel1.pdf). Continue with [Lab 1](02-lab-prompting.md) to compare the techniques on real dialogues.