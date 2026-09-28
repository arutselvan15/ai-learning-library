# Why Efficient LLM Deployment Matters

> Converted transcript. Timestamp markers removed for readability.

## Lesson Transcript

You may have heard a lot about training AI models, the massive GPU clusters, weeks of compute and billions of dollars. But the majority of AI cost is in running models, because training happens once, but inference happens every single time a user sends a message. So in this lesson, you'll see why efficient LLM deployment matters,

the explosion of open source models, the real cost of running them in production, and the two big categories of optimization, model and inference. So, let's dive in. In early 2023, there were essentially no competitive open source models. And now we have thousands, even millions on Hugging Face

from every organization, even OpenAI themselves. So, the challenge has shifted from, can I get a good model to, can I run that model efficiently? 

## Why Self-Host Models

Why would you want to run models yourself instead of calling an API? Well, there's four big reasons. Firstly, Cost Savings.

Because instead of paying per token, you can match model size to task difficulty. Secondly, is Security because data never leaves your environment. And that's quite important in industries like healthcare and financial services. Third is Control. Models can upgrade or deprecate when you say so, and there's no rate limits and no

issues with a third party being down. And finally, Customization. You can fine-tune models for accuracy and cost control. 

## Production SLOs

Now, like any production application, LLM deployments need measurable targets. We use service level objectives or SLOs to measure this. Now, there's two dimensions to track. First off is accuracy because a model that's

wrong isn't helpful. For example, a model that hallucinates or gives off-brand answers isn't useful. Your accuracy has to clear a usable threshold and that threshold depends on your use case. And this is where model cards come in. On the right is an excerpt from Llama 3.1's model card,

comparing the original 70B model against an optimized smaller version of it. Each row is a standardized benchmark. So you've got MMLU for general knowledge, GSM8K for math, and so on. That recovery column shows how much of the original accuracy, the optimized version retains 99.88% on average. And this is how you verify a

model meets your accuracy SLO before deploying it. Second is inference performance. And there's three latency metrics that really matter. So, the time to first token. This is the time taken to generate the first token of the output, which indicates how long the user is waiting before seeing any response.

Then you've got the inter token latency. That's the average time between generating consecutive tokens in the output, excluding the first token. And that helps assess the smoothness and speed of token generation. And finally, you've got the request latency, the total time from end to end. And don't forget about throughput.

Throughput is the average number of output tokens generated per second across all requests. In other words, can your system handle production level scale? See, small gaps in either dimension can mean big issues later on down the road. And to be useful, an LLM must

both be fast enough and accurate enough. 

## Hardware Requirements

Now, let's see what it actually takes to run them, starting with the hardware. GPU memory has to hold two things. First off is the model weights. These occupy a fixed space whether

you're serving one user or 100 users. And the KV cache. This is a working memory that grows with every token and every active request. We're going to be unpacking both in the next lesson, but for now, let's think about the hardware requirements to run an LLM.

For our example, we're going to take Llama 3 at 70 billion parameters. And the weights alone need about 140 gigabytes because a parameter is roughly about two bytes. So you need at least two 80 gigabyte GPUs just to load the model.

In practice, you would deploy it on four 80 gigabyte GPUs. That's the standard production unit for a 70B class model, giving you 320 gigabytes total, which leaves about 180 gigabytes for everything else. That everything else is the KV cache. For Llama 3 70B, one long context request

at 32000 tokens needs about 10 gigabytes in KV cache by itself. So with 180 gigabytes of headroom, you can serve 18 long-context users in parallel. However, a naive serving setup with no memory optimization can lead to poor memory management and reduce the number of users that can be served in parallel

from 18 down to just two or three. And that is the gap that we're closing. 

## Model and Inference Optimization

This course covers two categories of optimization to save on infrastructure requirements while speeding up inference for our users. You've got Model Optimizations and Inference Optimizations.

Model Optimizations are applied to the model itself before you even deploy it, using techniques like Quantization and Sparsification. The goal is to reduce the model's memory footprint and computational requirements while preserving as much accuracy as possible.

With Inference Optimizations, these happen at runtime in the inference engine itself. is where techniques like Continuous batching, Prefix caching and PagedAttention come in. And these don't change the model, but they change how efficiently you run it.

And don't worry if you're not familiar with these topics, you're going to be learning about them in this course. 

## Accuracy, Performance, and Cost

Now, here's a reality of deploying AI models. You're always going to be balancing between performance, accuracy, and cost. Performance could be measured in terms of latency and throughput. Because real-time latency is critical so your users aren't waiting around,

but that high throughput often requires more compute. At the same time, accuracy is needed for trust, but the larger models with perhaps higher evaluations tend to cost more. And we also need to keep infrastructure costs down. But significant model optimization can hurt accuracy.

So, most deployments have to pick two. However, the right tooling and techniques can push those limits, and that's exactly what this course is about. 

## Deployment Cost

Let's take a look at LLM deployment cost. With a naive LLM deployment of Llama where we load full precision weights,

and process one request at a time, it can be quite expensive. Hundreds of thousands to millions of dollars every month to serve the thousands of users that may be using the AI application. Now, let's add in Continuous Batching

and management of KV cache using PagedAttention with vLLM, a production inference server. You're batching multiple requests together. The throughput improvement here is dramatic. And instead of millions, we're saving

almost 10 times the cost. Finally, add model optimization. using the same vLLM deployment but with a quantized model. You've cut the memory footprint, sped up weight loading, and you're using the GPU's hardware more efficiently. This is where your mind might get blown. But before we dive into the model and inference optimizations,

the next lesson takes a step back to cover the inference and memory fundamentals. So what actually happens when a model generates a token, where the weights and KV cache live on the GPU, and how data moves inside the GPU. We'll see you there.

