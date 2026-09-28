# LLM Optimization Fundamentals

> Converted transcript. Timestamp markers removed for readability.

## Lesson Transcript

Now that you understand the GPU memory hierarchy and how inference actually works, let's put it to use. In this lesson, you'll see what's driving model sizes up and learn how compression makes even the biggest models runnable. Alright, let's dive in. As models have been getting more

capable, they've also been getting much larger. More parameters means more memory, more compute, and higher cost. So, this chart plots model size over time on a logarithmic axis. From the original Transformer in 2017 at 50 million parameters to today's frontier models in the hundreds of billions, some pushing past a trillion.

The key number here is that model sizes have roughly doubled every year. That's faster than how GPU memory is grown, which means that the gap between what models can do and what hardware can run keeps widening. And that's why compression is really important.

The chart on the right shows this gap. Model size has exploded while GPU memory has barely budged in comparison. 

## The Deployment Problem

And that mismatch creates four real problems. The first is with GPUs and infrastructure. Bigger models typically means more hardware accelerators, often spread across multiple nodes and that can get expensive fast.

Secondly is the tradeoffs that we have to make in user experience. because with more parameters, this may mean slower responses, lower throughput and less room for long context in the KV cache. Next is the energy and carbon footprint. because every extra GPU draws power and at scale that adds up to a real environmental cost.

And finally is the risk of model obsolescence. You risk investing in infrastructure for a model that gets superseded quickly. 

## Quantization

So, how do we close this gap? Compression techniques, in particular quantization, enable big models to run efficiently. and on less hardware. The idea behind quantization is straightforward. Think of the model weights being stored like pi,

which might be 3.14159, etc. Instead of storing that larger, more precise value, we store it as 3.14 using fewer bits. Most LLMs today are released at BF16. That's brain float 16 using 16 bits per number. Quantization converts those numbers into even lower bit formats

like FP8 or Float Point 8. INT8, Integer 8 or Integer 4. Fewer bits per number means a smaller model overall. 

## Precision Formats

Now, a quick note on naming. FP stands for Floating-Point. So, think about numbers with decimals like 3.14 BF is Brain Floating-point. BF16 is a 16-bit format

developed by Google with a wider range than floating-point 16, which makes it more stable for large models. And finally, INT stands for integers, like whole numbers, think of three or negative 127. And looking at the chart on the right, FP32 covers a huge range with fine-grained precision,

meaning tiny gaps between the representable values. BF16 covers the same range as FP32, but with less precision in between. And as you move down from FP16 and INT8, the range shrinks and the gaps between values get bigger. So that means you're trading precision for size.



## Weights and Activations

Finally, quantization can be applied to two things. Firstly, the weights, which are the model's learned parameters, and the activations, which are the intermediate values computed during a forward pass. We'll come back to that distinction later in the lesson. 

## Sparsification

In addition to quantization, sparsification can also be used

to zero out weights that contribute least to the model's predictions so they can be skipped entirely. during inference. A common approach is 2 to 4 sparsity where two out of every four values in a weight tensor are set to zero, reducing both memory and computation needed.

Together, quantization and sparsification reduce the model's memory footprint, making models easier to deploy on less hardware. And this means large models can run on fewer GPUs, directly translating to lower cost. 

## Memory Savings in Practice

So let's take the example of Llama 4 Scout

which comes in at 109 billion parameters. Its starting point is BFloat16, the format that the model ships in. At 16 bits per parameter, which is two bytes, the math is 109 billion times 2, which gives you about 220 gigabytes of just weights alone. To load that, you would need at least three 80 gigabyte GPUs.

Now, let's apply quantization. If we drop from BFloat16 down to INT8 or FP8, which is 8 bits per parameter or 1 byte. The math becomes 109 billion times 1. So those same weights now take 109 gigabytes, which is a 50% reduction.

And we go from needing three GPUs down to just two 80 gigabyte GPUs. Same model, but half the memory. If you go further with INT4 or FP4 quantization, you're down to roughly 55 gigabytes. That's a 75% reduction from the original.

And you've gone from three GPUs to just one, which is a huge infrastructure savings. So far, we've talked about quantization in the abstract, right? Fewer bits per number and a smaller model. But where exactly inside of the model does this happen? Remember from the last lesson that a

model is a stack of transformer blocks. And each block contains a Self-Attention block with four linear layers and a Feed-Forward Network with three linear layers. These linear layers are where we focus quantization. Note that this diagram doesn't show every layer or every weight in the model

because there are other components like the embedding layer at the start and the LM head at the end, but they're typically excluded from quantization to preserve accuracy. Quantization specifically targets the linear layers that are shown inside of the transformer block. And why these specifically? Well, that's because most of the time

during a forward pass is spent inside of these linear layers. It's where the main matrix multiplications happen, and it's where the bulk of the model's weights live. So that makes them the highest impact target. And there are two things to quantize inside a linear layer.

The weights and the input activations that flow through them. Input activations are simply the tensor that gets multiplied by the weights in every linear layer. Anything that gets multiplied by those linear weights inside of the Self-Attention and Feed-Forward Networks is what we call an input activation.

For example, inside attention, the input activation could be the tensor representation of fox that flows into the weights of the linear layers to produce Q, K, and V or it could be the weighted sum output that flows into O projection.

All of these are input activations. They're tensors moving through the network that get multiplied by weights along the way. Now that you know what gets quantized, which are the weights and input activations of linear layers, let's see how quantizing each can help with the LLM's performance.

Because quantization doesn't just save memory, but there are two effects that map directly onto the GPU memory hierarchy from the last lesson. First, Quantized weights means lower latency from data movement. Remember, every forward pass, the GPU pulls weights from the HBM into SRAM

so that the Tensor Cores can do the math. If those weights are 8-bits instead of 16-bits, there's literally half as much data to move. So it moves faster, which means faster inference. Secondly, quantized activations mean higher throughput via Tensor Cores. Remember that Tensor Cores are the specialized hardware

on the GPU that handles those matrix multiplications. and they can do more operations per second when the numbers are in lower precision formats. On modern GPUs like Hopper and Ada Lovelace and newer, there are dedicated FP8 Tensor Cores. On older Ampere GPUs,

INT8 Tensor Cores play the same role. So, weight quantization speeds up the data movement part of inference, and activation quantization speeds up the compute part. Quantizing both is what unlocks the full speedup. This gives you a choice when you're quantizing a model. Do you quantize just the weights or both the weights and the activations.

Let's compare the two schemes. For Weight-Only Quantization, as an example weight 8 activation 16 or W8A16, only the weights are quantized. For example, to INT8. The activations stay in higher precision like BF16. And at inference time, the weights get loaded

from HBM into SRAM in their compressed form. And then dequantized back to BF16 just before the multiplication. so that the Tensor Cores do the math in higher precision. The win here is purely on the data movement side. There's less data to pull from the HBM into SRAM, but you don't get the Tensor Core speed up.

For weight and activation quantization, as an example, W8A8, or weight 8 and activation 8, both the weights and activations are quantized. For example, to INT8 or FP8. At inference time, the math itself runs on lower precision Tensor Cores,

like the FP8 Tensor Cores on Hopper. So, you get both wins. You've got less data moving from HBM to SRAM, and Tensor Cores doing more operations per second. This reduces both the memory cost and the compute cost. So, what does this give you in practice? Well, there's five things. First is fewer GPU resources needed because

a quantized model fits on fewer GPUs, like you saw with Llama 4 Scout going from three GPUs down to one. Secondly is reduced deployment costs because with those fewer GPUs and smaller nodes, means a lower bill, whether you're paying for cloud instances

or running LLMs on your own hardware. Then, decrease latency. With less data moving from the HBM to SRAM means faster forward passes, which for your users means faster responses. Then, you've got higher throughput and longer context.

Because with more GPU memory freed up by smaller weights, you can fit more concurrent users in the KV Cache or serve longer context requests per user. And finally is lower energy consumption. With fewer GPUs running for less time means less power drain and a smaller carbon footprint at scale.

Let me show you these benefits in a real world use case. The example here is Retrieval Augmented Generation or RAG, where a model answers a question using documents pulled from a knowledge base. The input is 1024 tokens total, but it's broken into three pieces. 50 tokens for the system prompt,

something like answer using the provided context, 20 tokens for the user's question, like, what is the policy on and 900 tokens for the retrieved context. The documents or PDFs that contain the answer. The output is going to be approximately 128 tokens, which is the actual answer the

model generates using all of that information. So, this is a realistic long input workload where you have lots of context going in, a moderate response coming out, and exactly the type of scenario where quantization is going to pay off

because there's a lot of data to move and a lot of compute to do. Here are the results comparing FP16, which is the baseline model against FP8 where the weights and activations are quantized on Llama 3 at 70B running on two H100 GPUs. Take a look at the chart on the left. That's throughput.

So how many input tokens per second the system can process as you ramp up the user load. The FP16 baseline plateaus around 158 tokens per second, while FP8 climbs all the way to 474. That's over three times an improvement on throughput using this same hardware. The chart on the right is time to first token,

which is the latency before a user sees anything from the LLM. As the load increases, the FP16 base latency explodes, hitting over 30000 milliseconds. That's 30 seconds of waiting. At FP8, it stays much flatter, peaking around over 4800 milliseconds.

That's a 67 times reduction in latency at high load. The natural question at this point is, well, if you're throwing away the precision of the model, does the model get worse? Well, that answer when quantization is done correctly is essentially no. But done correctly matters here. Naive quantization where you're just rounding every number

down to a few bits does hurt the model. What works in practice are calibrated techniques that you're going to learn about, like GPTQ, AWQ, and SmoothQuant, which instead of rounding blindly, they use a small representative dataset to figure out which weights and values matter most

and protect those during the quantization process. We're going to go deeper into this and how it works in the next lesson. But for now, just know that that's how the results we're about to look at here were produced. The study evaluated quantized models on three reasoning benchmarks.

The AIME 2024 for competition math, which is a series of 30 expert level competition math problems. The MATH-500 for general math problems, which is a curated set of 500 challenging problems. And GPQA-Diamond for expert validated science questions. And the Y-axis on the right shows the metrics

which is average pass@1, which is the percentage of problems the model gets right on its first attempt. And it's averaged across all three benchmarks. So higher is better. Each cluster on the chart is a different model size from Llama-8B to the left to Qwen-32B on the right.

And within each cluster, the four bars represent four precision formats. Gray is BF16, original full precision model, and that's our baseline. Blue is FP W8A8, which is 8 8-bit floating point weights and activations. Green is INT W8A8. So 8-bit integer weights and activations.

And yellow is INT W4A16, which is the most aggressive scheme with 4-bit weights. The high level result that you can already see in the chart across nearly every model is that the four bars are essentially the same height. That's because quantization, done with these calibrated techniques,

doesn't meaningfully degrade accuracy. But let's zoom into a specific example. Let's take the Qwen-14B model. The original released BF16 model scores a 73.6 But the most aggressive quantization on the chart, which is INT W4A16 with those four-bit weights, scores a 72.8

That's a drop of less than one point while shrinking the weights by four times. Let's take the same model, but this time look at the blue bar. That's FP W8A8. So 8-bit floating point weights and activations.

Remember that original released model scored a 73.6 and the FP W8A8 quantized version scores a 74.3. How is that possible? The quantized model actually scored slightly higher? Well, that's not because quantization made it smarter. The difference is small enough to be random variation between runs, but it makes the point clearly. The 8-bit floating point

model precision where we shrunk the model by two times, will in some cases have no meaningful accuracy loss at all. So, the key takeaway across both examples is that done correctly, quantization gives you the speed and memory wins without giving up the model's quality. And in the next lesson, you'll see exactly how these techniques

like GPTQ and AWQ make this possible. And you're going to walk through the process of compressing a model yourself.

