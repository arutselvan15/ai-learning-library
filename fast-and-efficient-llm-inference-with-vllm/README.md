# LLM Inference Learning Guide

Catalog for the transcript-derived learning material. Start with a phase overview, then work through its chapters in order.

## Course Outcome

Build enough understanding to estimate inference memory, compress a model, operate vLLM, measure serving performance, evaluate quality, and make a deployment recommendation.

https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm

## Course Map

| Phase | Focus | Chapters |
| --- | --- | --- |
| 1. Foundations | Inference flow, transformer computation, memory, and deployment economics | [Phase overview](phases/01-foundations/README.md), [Chapter 1](phases/01-foundations/01-efficient-deployment/README.md), [Chapter 2](phases/01-foundations/02-inference-memory/README.md) |
| 2. Model optimization | Quantization, sparsification, calibration, and compression validation | [Phase overview](phases/02-model-optimization/README.md), [Chapter 1](phases/02-model-optimization/01-optimization-fundamentals/README.md), [Chapter 2](phases/02-model-optimization/02-llm-compressor-lab/README.md) |
| 3. Inference optimization | Continuous batching, PagedAttention, prefix caching, and vLLM | [Phase overview](phases/03-inference-optimization/README.md), [Chapter 1](phases/03-inference-optimization/01-vllm-serving/README.md), [Chapter 2](phases/03-inference-optimization/02-vllm-lab/README.md) |
| 4. Evaluation | SLOs, load patterns, latency percentiles, and model quality | [Phase overview](phases/04-evaluation/README.md), [Chapter 1](phases/04-evaluation/01-evaluation-benchmarking/README.md) |
| 5. Summary | Connect optimization, deployment, benchmarking, and final decisions | [Phase overview](phases/05-summary/README.md) |

## Learning Flow

```text
Foundations -> Model optimization -> Inference optimization -> Evaluation -> Summary
```

The course moves from understanding why inference is difficult, to changing the model, to improving runtime utilization, and finally to proving whether the result is deployable.

## Completion Checklist

- [ ] Calculate model-weight and KV-cache requirements.
- [ ] Produce a compressed checkpoint with size and quality measurements.
- [ ] Run a vLLM endpoint and inspect its metrics.
- [ ] Observe concurrency and prefix-cache behavior.
- [ ] Produce a benchmark report with latency percentiles and throughput.
- [ ] Compare quality with an uncompressed or published baseline.
- [ ] Write a deployment recommendation covering accuracy, performance, and cost.

## Labs

| Phase | Lab work | Notebook |
| --- | --- | --- |
| 1. Foundations | Memory and KV-cache calculations in the chapter exercises | No dedicated notebook |
| 2. Model optimization | [LLM Compressor Lab](phases/02-model-optimization/02-llm-compressor-lab/README.md) | [Open notebook](phases/02-model-optimization/02-llm-compressor-lab/lab.ipynb) |
| 3. Inference optimization | [vLLM Lab](phases/03-inference-optimization/02-vllm-lab/README.md) | [Open notebook](phases/03-inference-optimization/02-vllm-lab/lab.ipynb) |
| 4. Evaluation | [Evaluation Lab](phases/04-evaluation/01-evaluation-benchmarking/README.md) | [Open notebook](phases/04-evaluation/01-evaluation-benchmarking/lab.ipynb) |

## Course Summary

Review the [Phase 5 summary](phases/05-summary/README.md) after completing the evaluation work.

## Transcripts

Raw lesson transcripts are organized by phase and linked from the corresponding chapter pages.

| Phase | Transcript source |
| --- | --- |
| 1. Foundations | [Foundation transcripts](phases/01-foundations/README.md) |
| 2. Model optimization | [Optimization transcripts](phases/02-model-optimization/README.md) |
| 3. Inference optimization | [Serving transcripts](phases/03-inference-optimization/README.md) |
| 4. Evaluation | [Evaluation transcript and references](phases/04-evaluation/README.md) |

## Scope

This course focuses on inference fundamentals, compression, vLLM, and evaluation. The broader roadmap continues into Kubernetes GPUs, gateways, distributed inference, retrieval models, observability, multi-tenancy, and the capstone platform.