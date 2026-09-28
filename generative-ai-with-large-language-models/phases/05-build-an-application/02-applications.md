# Chapter 2 - Build an LLM-powered Application

[Previous: Chapter 1: Optimization](01-optimization.md) | [Contents](../../README.md) | [Quick reference](../../quick-reference.md)

## Match the intervention to the failure

An LLM can produce outdated answers, fail at arithmetic, or invent plausible-sounding facts. Changing prompts or fine-tuning is not always the right remedy. Use **retrieval** for information that changes or is private, a **code interpreter** for exact calculations, and independent checks for facts and safety.

A model's knowledge is bounded by its training data and cutoff. A model trained before a leadership change may confidently name the former office-holder. Hallucination is the related failure mode of generating a plausible answer when the model does not know, such as inventing a description of a nonexistent plant. Arithmetic fails for a different reason: the model predicts a likely token sequence rather than executing a reliable numerical operation, so route calculations to a verified tool.

![Examples of out-of-date, incorrect arithmetic, and hallucinated answers](../../assets/llm-apps/model-problems.png)

In retrieval-augmented generation (**RAG**), a retriever selects relevant passages from authorized documents and passes them as context to the model. For a question about employee parking, retrieve the current parking policy, then ask the model to answer using that passage and cite it. Retrieval quality and document trust matter: a retrieved page can be irrelevant, outdated, or contain instructions the application must not obey.

A vector store keeps embeddings of document chunks so a query embedding can retrieve semantically similar passages quickly. The application still needs chunking, metadata filters, access control, freshness checks, and an evaluation set for retrieval quality. Vector similarity is a ranking signal, not proof that a passage is authoritative or relevant.

![External data helps ground a model response](../../assets/llm-apps/solution.png)

## Tools and orchestration

Examples showing intermediate reasoning steps can improve some tasks, but generated reasoning may still be wrong. **PAL** asks a model to produce a program and uses an interpreter to compute the result; validate or sandbox any generated code. **ReAct-style** workflows interleave reasoning with allowed actions and observations, such as searching a source before answering. An orchestrator coordinates the model, retrieval, tools, and application interface; frameworks such as LangChain are optional implementations, not prerequisites.

For an operational workflow, an LLM might retrieve an order, confirm its items, call a shipping API through a constrained tool, and then request an email-label action. The completion should specify an allowed action, correctly formatted arguments, and the information needed to validate the result. ReAct makes this explicit as Thought-Action-Observation steps, but the action set must be predefined because a model can propose steps that the application cannot safely execute. Common orchestration components include prompt templates, memory, tool integrations, chains, and agents that select among tools.

![Program-aided language models route generated programs to an interpreter](../../assets/llm-apps/pat-5.png)

![An LLM application connects users, model, tools, and external information](../../assets/llm-apps/infra.png)

## End-to-end exercise

Design a dialogue-summary service: define required facts and prohibited disclosures; use the [Lab 1](../02-prompt-and-evaluate/02-lab-prompting.md) prompted baseline; compare [Lab 2](../03-adapt-a-model/02-lab-fine-tuning.md) tuned models on held-out dialogues; assess risks using [Lab 3](../04-align-behavior/02-lab-rlhf.md) as one illustrative signal. Add retrieval only if external policies are needed. Test prompt injection in retrieved text, wrong or missing facts, refusal behavior, latency, and cost before launch. Monitor after deployment because users and sources change.

**Checkpoint:** For a current HR policy question, would you choose RAG or task fine-tuning first? For an invoice calculation, what should the model do before answering? What should be logged or reviewed after deployment?

Sources: [model limitations](../../lecture-transcripts/knowledge-arithmetic-and-hallucination.md), [RAG](../../lecture-transcripts/rag-and-external-data.md), [reasoning examples](../../lecture-transcripts/reasoning-and-chain-of-thought.md), [PAL](../../lecture-transcripts/program-aided-language-models.md), [ReAct](../../lecture-transcripts/react-and-multi-step-tools.md), [application stack](../../lecture-transcripts/application-architecture.md), [responsible AI](../../lecture-transcripts/responsible-ai-considerations.md), [closing](../../lecture-transcripts/closing-and-research-directions.md); [Week 3 slides](../../slides/Week3.pdf).