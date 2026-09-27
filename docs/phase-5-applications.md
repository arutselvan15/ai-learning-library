# Phase 5: Optimization and Applications

[Study guide](README.md) | Previous: [Alignment](phase-4-alignment.md) | [Quick reference](quick-reference.md)

## In brief

An LLM application needs more than a model: an interface, orchestration, reliable information sources, evaluation, and a deployment plan. Optimization makes serving more practical; external tools or retrieval can address gaps in the model's knowledge or arithmetic. Each extra component creates a new failure mode to test.

## Learn in order

1. **Optimize for deployment.** The [application notes](../gen-ai/llm-apps/README.md) introduce distillation (train a smaller student from a teacher), post-training quantization (reduce numerical precision), and pruning (remove low-impact weights). The [compute notes](../gen-ai/model/compute/README.md) discuss precision and distributed training. Training-time precision choices and post-training quantization solve related but different problems. Always check quality and latency after optimization.
2. **Ground answers in external information.** The [RAG notes](../gen-ai/rag/README.md) show how retrieved documents can provide current or private context for an answer. Retrieval helps only when relevant, trustworthy passages are selected and the response stays grounded in them; it does not automatically eliminate hallucinations.
3. **Use tools for work the model should not guess.** The [application notes](../gen-ai/llm-apps/README.md) cover chain-of-thought examples, program-aided language models (PAL) for calculations, and ReAct-style reasoning and action. A code interpreter can calculate an exact result; a search or lookup tool can supply external facts. Tool calls need validation and appropriate permissions.
4. **Connect the system.** An orchestration layer coordinates the user interface, model, retrieval, and tools. Evaluate the whole workflow, including factuality, cost, latency, misuse, and failures, not just a single model response. The [project lifecycle](../gen-ai/project-lifecycle-and-roles/README.md) places integration and monitoring after model selection and adaptation.

## Putting it together

Sketch a dialogue-summary assistant for an organization:

1. Define what a good summary must contain and what data may be used.
2. Start with the Lab 1 prompted baseline; compare it to the Lab 2 fine-tuned or LoRA-backed model on held-out dialogues.
3. If the task needs private policy context, retrieve relevant documents and cite them in the answer. Do not let retrieved text override application instructions.
4. Test harmful, inaccurate, and incomplete outputs. The Lab 3 reward signal covers one risk, not the entire safety or quality review.
5. Measure response time and resource use. Consider quantization or a smaller model if the quality/cost trade-off supports it.

## Check your understanding

- When would you use RAG instead of fine-tuning to answer a question about a changing policy?
- How do distillation, quantization, and pruning differ?
- What needs testing when an LLM is allowed to call a code interpreter or retrieve external text?