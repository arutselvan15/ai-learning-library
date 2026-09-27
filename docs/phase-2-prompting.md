# Phase 2: Prompting and Evaluation

[Study guide](README.md) | Previous: [Foundations](phase-1-foundations.md) | Next: [Fine-tuning](phase-3-fine-tuning.md)

## In brief

Start with an existing model before training anything. Give it an explicit instruction and compare outputs with no examples (zero-shot), one example (one-shot), and several examples (few-shot). Control generation at inference time, then judge the outputs using both examples and task-appropriate metrics.

## Learn in order

1. Read the [prompting notes](../gen-ai/prompt/README.md). In-context examples are supplied in the prompt; they do **not** update model weights. Keep the context limit in mind.
2. Read the [generation settings](../gen-ai/generative-config/README.md). `max_new_tokens` caps the length of newly generated text; temperature affects randomness; top-k and top-p limit the candidates sampled for the next token. These settings change decoding, not the model's learned knowledge.
3. Read the [evaluation notes](../gen-ai/evaluation-metrics/README.md). ROUGE-1 counts overlapping unigrams, ROUGE-2 overlapping bigrams, and ROUGE-L uses a longest common subsequence. Precision measures overlap against generated text; recall measures overlap against a reference; their F1 balances the two. BLEU is commonly used for machine translation and relies on n-gram precision with a brevity penalty. A high overlap score can still hide a contradiction, so inspect examples as well.
4. Read the [benchmark notes](../gen-ai/benchmarks/README.md). GLUE and SuperGLUE assess language-understanding tasks; MMLU and BIG-bench probe broader capabilities; HELM considers multiple scenarios and metrics, including risks. Choose evaluations that match your use case, and guard against test-data contamination.

## Lab 1: Summarize dialogue

Open [Lab 1 notebook](../gen-ai/projects/summarize_dialog/Lab_1_summarize_dialogue.ipynb) and follow its sections in order:

1. Select the required kernel and install the dependencies specified inside the notebook. It checks for the original `ml.m5.2xlarge`-style environment (8 vCPUs and about 32 GiB RAM); a local machine may need adapted checks, packages, or more memory. [Local setup](../setup.md) is for separate API examples and is not a complete lab environment.
2. Load DialogSum and FLAN-T5. Compare a summary generated from a dialogue without an instruction to one using an instruction and then the FLAN-T5 prompt template.
3. Add one demonstration, then a few demonstrations. Compare what changes, and note the increased prompt length.
4. Change generation settings and inspect how the summary changes. Keep the dialogue and evaluation criterion fixed when comparing prompts or settings.

The [Lab 1 companion guide](../gen-ai/projects/summarize_dialog/READMD.md) links focused scripts for dataset loading, tokenization, prompting, and generation. The [Lab 1 lecture walkthrough](subtitles/subtitle%20%286%29.txt) revisits the exercise.

## What to take forward

Record a baseline summary, an improved prompt, and one failure case. If prompting does not reliably achieve the required task behavior, Phase 3 shows how task-specific training changes the model itself.

## Check your understanding

- What differs between one-shot prompting and fine-tuning?
- Why can ROUGE reward a summary that changes the meaning of the source?
- What should you hold constant when comparing two prompt designs?