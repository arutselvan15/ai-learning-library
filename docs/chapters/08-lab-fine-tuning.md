# 08. Lab 2: Fine-tune a Summarizer

[Previous: 07. Fine-tuning](07-fine-tuning.md) | [Contents](../README.md) | Next: [09. RLHF](09-rlhf.md)

## Goal and materials

Adapt FLAN-T5 for the same dialogue-summary task as Lab 1. Compare the original model, a fully fine-tuned model, and a base model with a LoRA adapter, using both generated examples and ROUGE. Use [the original Lab 2 notebook](../projects/fine-tuning/Lab_2_fine_tune_generative_ai_model.ipynb) as the exercise; [the lab lecture](../lecture-transcripts/lab-2-walkthrough.md) provides a narrated walkthrough.

## Follow the notebook in this order

1. **Set up:** Select the notebook's kernel, install its specified packages, and load the dataset and model. Its original instance check assumes an `ml.m5.2xlarge`-style 8-vCPU, 32-GiB environment. A different machine or updated package set may require adaptation.
2. **Baseline:** Test zero-shot dialogue summarization before training. Hold aside examples for comparison.
3. **Full tuning:** Preprocess input/target pairs, train or load the provided fine-tuned model, and inspect output against both reference and baseline.
4. **Evaluate:** Compute ROUGE on comparable examples and check for lost or invented details. Document the evaluation split.
5. **PEFT/LoRA:** Configure an adapter, train or load it as the notebook directs, then repeat qualitative and quantitative evaluation.
6. **Compare:** Record trainable parameter counts, checkpoint/storage implications, ROUGE, and whether sample summaries preserve the key facts.

Focused companion scripts, if useful: [load model](../projects/fine-tuning/load_model.py), [view parameter counts](../projects/fine-tuning/view_model_params.py), [full fine-tuning](../projects/fine-tuning/fine_tune_full.py), [LoRA](../projects/fine-tuning/fine_tune_peft_lora.py), [human review](../projects/fine-tuning/human_evaluation.py), and [ROUGE evaluation](../projects/fine-tuning/rouge_evaluation.py). These scripts are optional explorations, not prerequisites for running the notebook.

The notebook may download datasets, models, or checkpoints. The repository's [API example setup](../setup.md) is not a complete notebook environment, and saved notebook output should not be mistaken for a reproducible fresh result.

**Checkpoint:** Which model best balances summary faithfulness and resource usage for your chosen evaluation set? What evidence would make you change that decision?