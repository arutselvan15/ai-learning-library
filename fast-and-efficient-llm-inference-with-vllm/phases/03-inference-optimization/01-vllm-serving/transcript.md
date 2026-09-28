# Serving LLMs Efficiently with vLLM

> Converted transcript. Timestamp markers removed for readability.

## Lesson Transcript

So far, you've learned how to make models smaller, but shrinking the model is only half the story. How do you serve it to many users at once without the GPU sitting idle or running out of memory? In this lesson, you'll learn the three core techniques that make modern LLMs serve fast.

Continuous batching to keep the GPU busy, PagedAttention to help manage KV cache memory without waste, and prefix caching to skip KV recomputation when requests share content. And these are the techniques that power vLLM, the open-source inference engine you'll get hands-on with right after. Let's have some fun. 

## Inference Optimization

Now, let's shift to the inference

side and focus on the Inference Optimizations. These are the techniques that happen inside of the serving engine to maximize throughput and minimize latency. 

## Continuous Batching

Let's start with continuous batching. First, let's understand why batching matters. Think about how generation works, right? It's iterative. So every token requires a full forward pass,

and that means pulling all of the model's weights from HBM into the GPU's compute units. But here's the consequence, serving one request at a time leaves the GPU dramatically underutilized. The compute needed for a single token is tiny,

but you still pay the full cost of moving the entire model through memory. The tensor cores end up spending most of their time waiting for data and not doing math. This directly limits throughput or the number of tokens or requests that the system can process

per second across all users. So, the solution is batching, processing multiple requests together. Instead of reading the model's weights and using them for just one user, you read them once and use them for many users at the same

time. You've got the same memory cost, but you're getting much more work done per read. 

## Static Batching

The simplest way to batch requests is by doing static batching. You collect a fixed group of requests, process them together and wait until every single one finishes before starting the next batch. And this works well for traditional models like BERT or YOLO,

where the input and output sizes are predictable. A classification model that takes one image and returns one label has a fixed runtime per request. So, if you batch 10 images together, they all finish at roughly the same time. The GPU stays busy with this, but throughput is high.

Here's the thing though, LLMs break this assumption. So, take this example. Four requests start in the same batch, but they finish at different times. Request three finishes early at T5. Request one finishes early at T6,

and request two keeps going until T8. Once a request finishes, its slot in the batch sits idle until the longest request in the batch is done. And that GPU is wasting capacity on requests that are already complete. With static batching, a short request is stuck waiting for the long one.

And why does this happen with LLMs? Well, because the context length is unpredictable. One user might be asking, hey, what's 2 + 2 and get a five token answer. But another one might be asking for a 2000 word essay. And with Static Batching, the short request

is stuck waiting for the long one. And that's a lot of idle GPU time. 

## Continuous Batching in Practice

Continuous Batching solves this problem because instead of locking into a fixed batch and waiting for everyone to finish, the scheduler works at the token level.

The moment a request finishes, a new request immediately takes its slot in the batch. So here, as soon as request three finishes at T5, you'll notice that request five jumps in. And when request one finishes at T6, request six takes over. So that batch is never idle. Visually, you can see the difference.

Static Batching leaves GPU slots idle until the entire batch is finished, but Continuous Batching dynamically adds new requests to keep the GPU fully utilized. So far, we've talked about batching. to keep the GPU busy. But there's a second resource that limits how many requests you can serve concurrently.

That's the GPU's memory. And the biggest consumer of that memory is the KV cache. Remember from lesson two, every active request has its own KV cache, growing one token at a time. And the more users you serve, the more KV cache memory you need.

If you manage that memory poorly, you won't fit as many requests into a batch. So, throughput decreases even if the GPU has plenty of compute to spare. The KV cache is hard to manage because it has two properties.

First, it grows and shrinks dynamically, and secondly, you don't know in advance how long a request is going to be. Some users have those five token answers, but others get 2000 token essays. This diagram shows how earlier systems handled this.

For request A, the system pre-allocated one contiguous block size to the maximum possible length, 2048 slots. The yellow slots hold the prompt and the blue slots hold the tokens generated so far, plus a few reserved for what comes next. But take a look at this long stretch of empty slots after that.

2048 slots that will never be used. That wasted space inside of the allocation is what's called internal fragmentation. Now, take a look at the gap between request A and request B. These slots are physically free, but too small to fit a new request

pre-allocated chunk. So, they sit unused. That's what's known as external fragmentation. So it's wasted space between these allocations. And there's a third kind of waste. Even the slots request A will eventually use sit reserved and empty for most of its lifetime,

blocking other requests from using that space in the meantime. The paper that led to vLLM reports that only 20 to 40% of KV cache memory was actually used to store real tokens. And the rest of that was lost to fragmentation and over-reservation. So, the GPU has plenty of memory in theory,

but in practice, most of it is locked up. That limits how many requests can fit in a batch, which limits throughput. 

## PagedAttention

And vLLM's PagedAttention was designed to solve this. With PagedAttention, the core innovation introduced by vLLM, instead of storing the KV cache as one large continuous block,

we break that into fixed size blocks, also called pages. Each block holds the keys and values for a small number of tokens and the system keeps a block table that maps each request's token to the physical blocks holding them. idea is borrowed from virtual memory and

paging in operating systems. So, when your computer runs a program, it doesn't reserve one giant chunk of continuous RAM. Instead, the OS splits memory into small pages and scatters them wherever there's room. And it uses a page table to keep track of where everything is.

PagedAttention applies the same trick to the KV cache. So you have lots of small blocks scattered across GPU memory and stitched together by a lookup table. Let's walk through how this works step by step. Let's assume that we're starting with empty physical blocks in GPU memory that are waiting for a request.

The prompt "Artificial Intelligence is" comes in, and the system grabs one free block. Here is Block 3. And it stores the KV cache for the three prompt tokens. So, the Block table now records, hey, physical block 3 with three slots filled. The model then generates the next token, which is the.

Block 3 still has one empty slot, so we use that. And the Block table updates to four slots filled. There's no new block needed in this scenario. Now the model generates future. But block three is full, so the system grabs another free block, which is block six,

and it stores future there. Notice the new blocks don't have to be next to each other in memory. Block three and block six are physically separate and that's fine. The model then continues generating the next tokens of, then technology

and each token fills the next slot in block six. Memory is allocated only as the request needs it, and it fills one block at a time, never more. Now, let's see how attention works. To generate the next token, the model uses a query for the current token, here being technology,

and it needs to attend to the keys and values of all previous tokens. The system reads Block 3 from the Block table, fetches it from GPU memory and computes attention against those tokens. And it does that one block at a time. Then it does the same for Block 6.

The model attends to all previous tokens even though they're stored in non-contiguous memory blocks. The block table is what makes this stitching possible. And now two requests can share the same physical memory pool with each request blocks scattered wherever there's space. There's no pre-allocation, there's no wasted slots,

and most importantly, no fragmentation. Both requests get exactly the memory they need when they need it. And that's how vLLM fits more requests into the same GPU and pushes throughput up. 

## Prefix Caching

Here's another powerful optimization, prefix caching. When requests share the same prefix like a system prompt,

they share KV cache blocks. Compute once and reuse across users. So, Prefix Caching reuses the KV cache when requests share the same starting tokens instead of recomputing them from scratch. And there's two common patterns.

The first is shared prompts across users, which is shown on the left. So, say for example, three users send different questions, but they all hit the same system prompt. Well, without Prefix Caching, that prompt's KV cache would be recomputed

for every user. With it, the prompt is computed once and reused for everyone. And the same applies to few-shot examples or shared RAG context. The second is multi-turn conversations shown on the right. Prompt two's prompt includes everything from round one plus a new question. And since the round one part is identical,

its KV cache is pulled straight from memory instead of being recomputed. This model only does new work on the new tokens. This benchmark shows the impact, because as the cache hit rate climbs, throughput climbs with it. at a 75% hit rate, its throughput is roughly four times higher.

That's compute that the system simply doesn't have to redo. All of these techniques like continuous batching, PagedAttention and prefix caching come together in vLLM, the open source inference engine built to be the fastest and easiest to use. And the numbers showed that it landed. By January 2025, vLLM was seeing 100000 daily installs.

Usage grew 10 times in 2024, and it's one of the top AI and ML repositories on GitHub. by contributor count. So, the community momentum here is real. And vLLM is designed to work across the entire landscape. So any model like Llama, Qwen, DeepSeek, Gemma, Mistral, Granite and more are supported

across any hardware accelerator. So think of NVIDIA GPUs, AMD Instinct, Intel Gaudi, Google TPUs, AWS Neuron, IBM Spyre, and you can deploy this on any development environment across edge, private cloud, and public cloud.

So you have one platform for all of it. In the next lesson, you'll launch your own vLLM server and see continuous batching, PagedAttention and prefix caching live in the metrics. So, let's go serve a model.

