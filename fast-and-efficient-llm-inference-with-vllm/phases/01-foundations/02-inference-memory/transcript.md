# Inference Memory Fundamentals

> Converted transcript. Timestamp markers removed for readability.

## Lesson Transcript

Before we dive into the optimization techniques, let's take a closer look at what actually happens during inference. The computations involved in generating each token, what the KV cache is and why it matters, and how the GPU memory hierarchy shapes everything. These fundamentals will set you up for everything that comes next.

All right, let's go. 

## What Inference Is

When you interact with an AI application and send a prompt like asking a question or getting a code suggestion or summarizing document or maybe even have your AI agent do it for you these days. What's actually happening under the hood is called Inference,

which is the process of using a trained model to generate a response. Now, to actually run Inference in production, you need more than just the model files sitting on a disk. You need a stack of three pieces working together.

So at the top, the model itself. That's the file containing the billions of learned parameters like Llama or Qwen. In the middle, we've got the inference server. Software like vLLM that loads the model, manages incoming requests and handles all the

inference optimizations we're about to learn about. And at the bottom, the hardware accelerator, typically a GPU that does the heavy numerical lifting. Now, a quick note, because technically, you can run a model directly on a GPU using a library like PyTorch without an inference server in the middle.

That works fine for a notebook or a single user, but the moment you need to serve many users at once, efficiently is what this course is about. The inference server becomes essential. It's the piece that makes the GPU actually usable at production scale, and you'll see why exactly in the coming lessons,

but when a user sends a prompt, it travels down this stack, gets processed, and a response travels back up. 

## Autoregressive Generation

So, how does an LLM actually generate a response? LLMs don't produce a whole sentence at once, they actually generate one token at a time where a token can

be roughly a word or a piece of a word. So, here's how the loop works. So, the user sends a prompt like The quick brown, and the model processes that input and predicts the next token, which in this case would be fox. That new token gets appended to the input, so the

model's input is now the quick brown fox. And it runs again to predict the next token, which would be jumps. Then jumps is appended and the model runs again and predicts over. This continues until the model generates a special

end of sequence token that signals that it's done. And this is called autoregressive generation. Each new token depends on all of the tokens before it, including the ones that the model just generated. Every token in a response requires a full pass through the model.

So a 500 token answer means the model runs 500 times. 

## Transformer Forward Pass

So what actually happens during one of those forward passes? Well, let's take a look inside of the model. Because when the user sends the prompt, the quick brown, the model first converts each token into a series of numbers called a token embedding.

Those embeddings then flow through a stack of transformer layers, and each layer has two main parts. a Self-Attention block and a Feed-Forward Network. The Self-Attention is where tokens exchange information with each other, and the Feed-Forward Network then processes each token's representation further.

This pair, attention plus feed forward, repeats N times stacked on top of each other. After the final layer, the output goes through one last component called the LM Head, which turns the model's internal representation into a score for every possible next token.

The highest scoring token is the prediction. In this case, fox with 90% probability. Then, as you saw, that token gets appended to the input and the whole stack runs again for the next forward pass. Let's zoom into one of these

transformer layers to see what's actually inside. The Self-Attention block and the Feed-Forward Network are both built from linear layers. A linear layer is just a matrix multiplication. So it takes a vector of numbers, multiplies it by a weight matrix,

and produces a new vector. That's all. But this simple operation is where almost all of the model's parameters live, and where almost all the computation happens. Inside the Self-Attention block, there are four linear layers. Q, K, V, and O projections.

This is where tokens look at each other to figure out how they relate. Inside the Feed-Forward Network, there are three linear layers. gate, up, and down projections. These process each token's representation independently, without any interaction between tokens.

These linear layers are weight matrices. They're part of the model learned once during training. Everything the model does like understanding tokens, relating them to each other and predicting the next one, comes down to these linear layers doing matrix multiplications. Over and over across N transformer blocks

for every single token generated. 

## Self-Attention and KV Cache

We just said that self-attention is built from four linear layers. Q, K, V, and O projections. But let's see what they actually do to generate the next token after Fox. In self-attention, each token in the sequence pays attention

to the other tokens. For example, when the model reads the quick brown fox, attention is what lets fox know that it's connected to quick. The fox is quick. To do this, three vectors are first computed for the fourth token being fox.

The first one is Q, the query, which means what do I want to know from the context. The next one is K, the key. So it means here's my label and the kind of information I contain. And finally, V the value.

which means if my label matches, here is my actual content. These three vectors come from passing the tokens vector representation through three separate linear layers. Q, K, and V projections. The vector representation for fox

could be the input vector embedding of fox if this is just the first Transformer layer. Or it could be the output vector from the previous layer. Now, to compute attention for fox. First, we take its query, Q, and compare it against the key

of every token so far including itself using a dot product. A high dot product means this token is relevant for me, and a low dot product means not relevant. That result gives us four raw scores. We divide those scores by the square root of the key dimension

to keep the numbers stable and pass them through a softmax to turn them into weights that sum to one. In this example, token two gets the highest weight 0.52, meaning the model thinks token 2 is most relevant to token 4. Finally, we take a weighted sum of

all of the value vectors using those weights. The result is a single vector, now enriched with context from the rest of the sequence. That vector then passes through o_proj, the fourth linear layer, and that produces the final output of the attention block. There are two things to note here.



## KV Cache

First, to generate a new token, we need the keys and values of every previous token in the sequence. This query is only needed for the current token, but K and V are needed for the entire history. And that simple observation is the foundation of KV Cache.

So, let's walk through it. After the model generates the fourth token fox, it computes its query, key, and value. vectors. So Q4, K4, and V4. Attention then uses Q4 together with the keys and values of all previous tokens.

Now the model generates the next token jumps. For this token, it only needs a new query Q5 along with a new K5 and V5. The Ks and Vs of tokens 1 through 4 haven't changed since the last step. So recomputing them would be

wasteful. So, we cache them. After we compute the K and V for a token, we save them in GPU memory. And so on every inference step, we only compute K and V for the new token and append them to the cache. This is the KV cache.

And for attention, we get the Ks and Vs of all previous tokens from the cache. And remember, this is happening at every one of the N layers, so the savings multiply by N. Let's think about how big this cache can get. And to do that, we're actually going to compute it.

For every single token, we need to store Key and Value vectors at every layer. And here's one detail worth mentioning. Each layer actually computes several parallel sets of K and V per token to help the model capture different types of relationships between tokens.

These are called the KV heads, and in Llama 3 70B, that number is eight. So, all of this needs to be cached. And the formula to do this is two times the number of layers, times the number of KV, times the head dimension times the dtype_bytes.

Now, let me explain how that works. The two is because we store both K and V. For Llama 3 70B, the number of layers is 80. There are eight parallel K and V sets, the number of KV. The head dimension is 128, and in the precision that it was released at,

each number takes two bytes. So, when we plug in two times 80 times 8 times 128 times two, that's about 320 kilobytes per token. And that's per token. Let's scale it up to the context lengths that people actually deploy at.

So for example, a 2000 token context, which is a typical chat turn, is about 640 megabytes of cache. An 8000 token context, which is the standard production tier, is about two and a half gigabytes. A 32000 token context, which is a long document or code base,

is about 10 gigabytes. and 128000 token context, which is Llama 3's max is about 40 gigabytes. Now, look at that last number. The model weights at its release are about 140 gigabytes. And a standard 128000 token context request needs about 40 gigabytes of KV cache on top of that.

nearly a third of the model's own size just for one user's conversation. And this is per request. If you serve 10 concurrent long context users, you need over 400 gigabytes of KV cache alone on top of the model. And this is why the KV cache is the dominant memory concern in modern LLM inference.

It lives in GPU memory, it grows linearly with the sequence length and the number of concurrent requests, and managing it efficiently is the single biggest job of a production inference server. You've now seen the two main things that sit in memory during inference.

You've got the model weights and the KV cache. 

## GPU Memory Hierarchy

But let's zoom out and look at the memory system that they actually live in. First, A quick piece of vocabulary. So far you've seen words like vector, matrix, and intermediate values somewhat loosely.

The general term for any multi-dimensional array of numbers is called a tensor. A single number is a tensor with zero dimensions. A vector is a one-dimensional tensor, and a matrix is a two-dimensional tensor. Q, K, V, the KV cache,

and the model's weights are all tensors. So, from here on out, when we talk about data, moving through the model, all we mean is just tensors. Now, where do these tensors live? A GPU has three tiers of memory and they differ dramatically in both size and speed.

At the bottom here sits the CPU DRAM, the host machine's main memory. Because a GPU isn't a standalone computer, right? It plugs into a host machine with its own CPU and RAM. And the host memory varies a lot because your laptop might have 16 gigabytes, but a data center

server might have a terabyte or more. But whatever the size may be, DRAM is far from the GPU. And moving data from the host to the GPU is slow compared to anything happening on the GPU itself. In the middle sits HBM or high bandwidth memory. This is what people usually mean

when they say GPU memory or VRAM. This HBM is on the GPU card itself. It's close to the compute units. Now, it's smaller than the host memory, but it's still much faster. Up here at the top sits SRAM. It's on-chip memory that sits right next to the GPU's compute units.

Those compute units are called tensor cores. They are specialized hardware that performs matrix multiplications extremely fast, which is exactly what each linear layer in the model needs. The tensor cores read their input data from SRAM. They do the math, and they write the results back. SRAM is tiny, but it's extraordinarily fast,

which is what keeps the tensor cores fed. To put some numbers on this, let's look at one specific GPU, the NVIDIA A100. Its SRAM is about 20 megabytes, but delivers roughly 19 terabytes of bandwidth. Its HBM is 40 gigabytes at 1.5 terabytes per second.

And data transfers between the host and the GPU happen at only roughly 12 gigabytes per second. So, the tradeoff is clean. The closer memory sits to the compute units, the faster it is, but at the same time, the less of it that there is. as well. Now, where do our inference tensors live

and how do they move between tiers? So the model weights get loaded from the disk or the host DRAM into HBM one time at startup. And they stay there for the lifetime of the server. The KV Cache also lives in the HBM and remember, it grows as each request processes more tokens.



## Data Movement During Inference

So during every forward pass, small chunks of the weight and the KV Cache get pulled from the HBM into SRAM so that the tensor cores can do the math on them. This happens over and over for every linear layer at every one of the N transformer layers

for every token generated. There are also transient tensors produced at each step, like the Q, the K, and the V vectors for the current token, the attention output and the output of each feed forward layer. And this is all computed inside of the SRAM,

but it's used immediately by the tensor cores and then discarded. So, they don't persist in the SRAM. That means there's two things that govern how fast inference can happen. First is how fast data can move from the HBM into SRAM, and second is how fast the tensor cores can compute on that data once it arrives.

Every optimization you'll look at in the coming lessons comes back to this same principle. You need to move less data, move it more efficiently, or manage that memory better. In the next lesson, you'll take your first step into optimization. You'll look at quantization. It's a technique that shrinks the model

by storing its weights in lower precision formats. The result of that is less data to move through that memory hierarchy. So, we'll see you there.

