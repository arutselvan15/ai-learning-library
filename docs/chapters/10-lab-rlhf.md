# 10. Lab 3: Generate Less-toxic Summaries

[Previous: 09. RLHF](09-rlhf.md) | [Contents](../README.md) | Next: [11. Optimization](11-optimization.md)

## Goal and materials

Start with a summarization-tuned FLAN-T5 model and measure how PPO with a hate-speech classifier reward changes its summaries. Follow [the original Lab 3 notebook](../projects/rlhf/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb) and consult [the lab walkthrough transcript](../subtitles/subtitle%20%2830%29.txt) for the course explanation. This lab is conceptually downstream of Lab 2 but loads its own models; do not assume one notebook shares kernel state or saved artifacts with another.

## Follow the notebook in this order

1. **Set up:** Select the requested kernel and install the notebook's dependencies. The original course check expects 8 vCPUs and about 32 GiB RAM. Model downloads and training may be expensive; adapt to your environment rather than assuming [local API setup](../setup.md) suffices.
2. **Load roles:** Distinguish the policy generating summaries, the reference policy used for comparison, and the hate-speech classifier acting as a reward model/toxicity evaluator.
3. **Measure a baseline:** Record classifier-based toxicity before training, along with representative summary text and source dialogues.
4. **Tune with PPO:** Configure the PPO trainer and PEFT adapter as shown, then update the policy using reward scores. The classifier is a proxy for one type of unwanted language, not human judgment of overall quality.
5. **Compare:** Re-evaluate toxicity after training on comparable examples; read the summaries to see whether they still retain the dialogue's important facts. Look for both false positives and false negatives in the classifier.

Notebook outputs saved in the repository may come from an older course environment. Use them to understand expected shapes of results, not as evidence of a fresh run. Do not use one average toxicity score as a deployment safety claim.

**Checkpoint:** Describe a case where the toxicity metric improves while the summary becomes less useful. What additional human evaluation would reveal it?