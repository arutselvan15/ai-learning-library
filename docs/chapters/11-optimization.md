# 11. Optimize a Model for Serving

[Previous: 10. Lab 3](10-lab-rlhf.md) | [Contents](../README.md) | Next: [12. Applications](12-applications.md)

## Pick an optimization for the constraint

Serving has different constraints from training: response latency, memory, throughput, cost, and acceptable loss of quality. Measure a baseline first. These techniques change different things:

| Technique | What changes | What to verify |
| --- | --- | --- |
| Distillation | Train a smaller student to imitate a teacher's outputs or probability distribution | Task quality, transfer to hard examples, serving cost |
| Post-training quantization | Store or compute with lower-precision weights, sometimes activations | Accuracy, calibration requirements, hardware support, latency |
| Pruning | Remove or zero low-impact parameters/connections | Actual speed/memory gains on the target hardware and quality |

![Distillation, quantization and pruning compared](../assets/llm-apps/optimizing.png)

![A student model learns from a teacher model](../assets/llm-apps/distilation.png)

Post-training quantization may require representative data to calibrate activation ranges. Low precision often saves memory, but its exact speed benefit varies by hardware, kernels, and model architecture. Pruning only yields practical speedups when the runtime exploits the resulting sparsity or smaller structure. Distillation is not limited to encoder models and can produce a different, smaller model.

![Post-training quantization reduces weight precision](../assets/llm-apps/quantation.png)

For training a model too large for one GPU, revisit [distributed and sharded training](03-pretraining-compute.md); it solves a different problem from post-training inference optimization. Compare model quality, cost, and latency together after any optimization.

**Checkpoint:** Would quantization alone solve missing facts in a summary? Would FSDP alone make an already-trained model faster to serve? Explain which problem each tool actually addresses.

Sources: [deployment optimization lecture](../subtitles/subtitle%20%2831%29.txt), [lifecycle effort](../subtitles/subtitle%20%2832%29.txt); [Week 3 slides](../slides/Week3.pdf).