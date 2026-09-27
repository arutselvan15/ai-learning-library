# Phase 4: Alignment with Human Feedback

[Study guide](README.md) | Previous: [Fine-tuning](phase-3-fine-tuning.md) | Next: [Applications](phase-5-applications.md)

## In brief

A model can follow instructions and still produce unwanted responses. RLHF uses human preferences to define desired behavior, trains a reward model from those preferences, and uses reinforcement learning to update the language model. The lab uses a pretrained hate-speech classifier as its reward signal to make summaries less toxic; it is a focused illustration of the alignment workflow, not a guarantee of safety.

## Learn in order

1. Read the [RLHF notes](../gen-ai/rlhf/README.md). Helpful, honest, and harmless are separate objectives that may trade off. Human raters can rank alternative responses to the same prompt; those preferences can become training pairs for a reward model.
2. Understand the feedback loop: a policy (the LLM) generates responses, a reward model scores them, and an RL optimizer such as PPO updates the policy toward higher reward. A reference policy and a KL penalty can discourage drifting too far from the starting model.
3. Separate **reward** from **real-world quality**. A toxicity classifier measures a particular kind of harmful language, not factual accuracy, completeness, or all types of harm. A system can optimize its score and still produce a poor summary.

## Lab 3: Detoxify summaries

Open [Lab 3 notebook](../gen-ai/projects/rlhf/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb). The [short lab guide](../gen-ai/projects/rlhf/README.md) and [lecture walkthrough](subtitles/subtitle%20%2830%29.txt) give the context for this exercise.

1. Prepare the notebook kernel and dependencies. Its original setup checks for an 8-vCPU, approximately 32-GiB instance; models and datasets may require downloads. Use the notebook's own installation and loading instructions.
2. Load the instruction fine-tuned summarization model and a hate-speech classification model. Note the roles: one generates summaries, the other supplies a reward/toxicity measurement.
3. Evaluate toxicity before optimization. Keep the evaluation examples available for a before/after comparison.
4. Configure and perform PPO-based training with PEFT, then evaluate toxicity quantitatively and inspect generated summaries qualitatively.
5. Discuss trade-offs: did toxicity change, and did the model retain useful, faithful summaries? Inspect edge cases, not just the average score.

The course walkthrough connects Lab 3 conceptually to Lab 2; follow the model-loading cells in Lab 3 rather than assuming artifacts or variables from a prior notebook are in memory. Notebook outputs may be saved from an earlier course environment and are not proof that a fresh run will reproduce the same result.

## Check your understanding

- How are human preferences converted into a reward signal in a typical RLHF pipeline?
- What is the difference between the policy model and the reward model in this lab?
- Why might optimizing a hate-speech score lower measured toxicity without ensuring high-quality summaries?