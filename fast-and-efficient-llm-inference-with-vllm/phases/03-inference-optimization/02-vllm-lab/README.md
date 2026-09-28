# Chapter 2 - vLLM Lab

## Purpose

Operate a local inference server and observe its runtime behavior.

Source transcript: [vLLM lab](transcript.md)

## Notebook

Work through the [vLLM lab notebook](lab.ipynb).

## Starter Command

```bash
vllm serve <model-name> \
  --dtype bfloat16 \
  --max-model-len <context-window>
```

## Starter Client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.completions.create(
    model="<served-model>",
    prompt="What is PagedAttention?",
    max_tokens=30,
)
print(response.choices[0].text)
```

## Endpoints

- `/v1/models`: readiness and served model identity.
- `/v1/completions`: completion requests and optional log probabilities.
- `/metrics`: running requests, waiting requests, token counters, and KV-cache usage.

## Practice

- Run one request and record latency.
- Send five concurrent requests and observe running and waiting counts.
- Send five requests with the same system prompt and different questions.
- Compare cache metrics and token throughput before and after repeated prefixes.

## Checkpoint

Connect observed metrics to batching and caching behavior.

Next: [Phase 4 - Evaluation](../../04-evaluation/README.md)