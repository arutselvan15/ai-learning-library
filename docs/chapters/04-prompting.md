# 04. Prompting and Generation

[Previous: 03. Pre-training](03-pretraining-compute.md) | [Contents](../README.md) | Next: [05. Lab 1](05-lab-prompting.md)

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

Keep the same dialogue and evaluation criteria when comparing prompts. For a summary, check who said what, important actions, omissions, and invented details. A natural-sounding but unfaithful summary is not a success.

**Checkpoint:** Write a zero-shot summarization instruction, then add one example. What extra tokens does the demonstration cost, and what behavior do you expect it to clarify?

Sources: [in-context prompting lecture](../subtitles/subtitle%20%283%29.txt), [generation controls lecture](../subtitles/subtitle%20%284%29.txt); [Week 1 slides](../slides/Weel1.pdf). Continue with [Lab 1](05-lab-prompting.md) to compare the techniques on real dialogues.