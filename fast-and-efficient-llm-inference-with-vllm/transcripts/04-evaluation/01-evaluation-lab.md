# Evaluation Lab

> Converted transcript. Timestamp markers removed for readability.

## Lesson Transcript



## Why Measure

Now, let's shift to the evaluation side. You've optimized a model and you're serving it, but how do you actually know if it meets your requirements? Is the deployment fast enough and are the responses good enough? You'll be doing benchmarks with GuideLLM for performance under load and evaluations with lm_eval for quality measurement.

They're both open source tools. So, let's go. We keep coming back to this triangle because delivering production-ready LLMs means navigating tradeoffs between accuracy, performance, and cost. In practice, you can optimize for any two corners, but the third one will pay the price. So, high accuracy with low latency means high cost.

Low cost with high accuracy means high latency. And low cost with low latency means you're sacrificing accuracy. So, where does your deployment need to land on this triangle? Well, the thing is, without clear measurements, you can't always answer that question. And that's what this lesson is about.



## Evaluation and Benchmarking

The answer is measurement. and there are two complementary kinds. Model Evaluation is the broader process, assessing whether a model is fit for purpose across criteria like accuracy, safety, and task suitability. Model Benchmarking sits inside that.

It's the standard comparison of a model against predefined data sets, tasks, or other models using objective metrics. Think of evaluation as the question, is this model good enough for what I need? and benchmarking as one the tools you use to answer it. 

## Define SLOs

Before you benchmark anything, you need to know what you're benchmarking against.

And that's where SLOs or service level objectives come in. Let's take a look at two common use cases and the very different targets they imply. First with an E-commerce Chatbot. This lives or dies on responsiveness because users expect conversational speed,

so we set tight targets. A Time To First Token under 200ms and Inter-Token Latency under 50ms. seconds. And we hold those at the 99th percentile, meaning 99% of requests must meet them. Now, a RAG system is a different type of beast.

Users are willing to wait a bit longer because they're expecting a thoughtful and grounded answer. So, we relax these targets. 300 milliseconds for the time to first token, 100 milliseconds for inter-token latency when streamed, and end-to-end latency under 3 seconds.

So, we have same metrics, but very three different thresholds for them. And the lesson here is simple: define your SLOs before you benchmark, because the numbers only mean something relative to your targets. 

## Performance Benchmarking with GuideLLM

Now, instead of writing our own benchmarking tool, we're going to use GuideLLM.

It's an open-source tool from the vLLM project, and it puts your inference server under controlled load and measures what comes back. What makes GuideLLM useful is that it's purpose-built for LLM serving. Generic load testers measure request latency as one number, but GuideLLM understands streaming responses.

So it captures the metrics that actually matter for your SLOs, like the time to first token, the inter-token latency, and more. You can run it ad hoc from the command line or wire it into CI or continuous integration to catch regressions automatically. 

## Benchmark Use Cases

There are four scenarios where benchmarking pays off,

and you'll likely hit all of them at some point in a deployment's life cycle. So let's walk through them. First, is pre-deployment. Before you commit to a model, you need to know if it'll actually work on your hardware at the quality level you need.

For example, on an NVIDIA H200 GPU, should I use Llama 3.1 8 billion parameters or 70 billion parameters for a customer service chatbot? The bigger model is more capable, but can your hardware serve it with your latency budget? Benchmarking answers that before you're stuck with the wrong choice in production.

Second, cost and capacity planning. Once you've picked a model, the next question is how much hardware you need. So you might ask how many servers do I need to keep my service running under peak load. Benchmarking tells you the throughput per server,

which feeds directly into how many you need to provision and how much it'll cost. Third is Regression &amp; A/B Testing. Because models change and quite frequently. You'll quantize them, you'll swap versions, you'll tune the serving configuration and each change can shift performance.

in ways that aren't always obvious. Benchmarking lets you ask things like how much more traffic can the INT8 version handle compared to the baseline and catch regressions before users do. And finally, Hardware evaluation. So, let's say you want to

ask what's the maximum request per second my hardware can handle before performance starts to degrade. This is how you find the breaking point, the load level where latency starts to climb sharply. Knowing that number is critical for autoscaling. and setting honest capacity limits for

the LLMs that we want to serve. Each of these requires running realistic benchmarks under real load patterns, and you can't really guess your way through these. 

## Workload Patterns

So, how do we simulate realistic load? GuideLLM gives you five traffic patterns

and each one tells you something different about your deployment. Synchronous runs one request at a time, waiting for each to finish before sending the next. And this is your clean baseline. It's single request latency with no queuing. Concurrent runs a fixed number of parallel streams,

and this shows how the server holds up when multiple users are hitting it simultaneously. Constant sends requests asynchronously at a fixed rate that you specify. And it's useful for simulating steady and predictable traffic. Poisson also sends at a rate that you set but with random spacing that follows a Poisson distribution.

And this is the closest match to real user traffic where requests are going to be arriving unpredictably. And sweep runs the whole spectrum automatically. Synchronous as the floor, concurrent as the ceiling, with several constant rate runs in between. And you get a full performance curve in one go,

and it's great for capacity planning. One thing to keep in mind, the benchmark numbers you get are always shaped by everything else in the stack. The model architecture and size, whether it's quantized, the serving engine you're using, the hardware, and your batching settings, just to name a few.

The benchmark doesn't change these variables, it just measures their combined effect. 

## Quality Evaluation

Performance is only half the picture. A model that's blazing fast but gives wrong answers isn't useful. And a quantization technique that wins on throughput but also affects accuracy isn't a win at all. You've probably seen claims like, hey, this is our smartest model yet.

But how does anyone actually verify that? The answer is standardized accuracy benchmarks. Tests like MMLU, HellaSwag, and GSM 8k, that run across many models so the results are directly comparable. And that's what produces leaderboards like the Artificial Analysis Intelligence Index that's shown here.

The tool that we'll use for accuracy evaluation is lm_eval. The LM evaluation harness from Eleuther AI. It supports a huge range of standardized benchmarks out of the box, covering general knowledge, reasoning, math, coding, and more. What makes it especially useful is that it works with both local models and remote API endpoints,

including the vLLM server that we already have running. So, you can evaluate your optimized model on public benchmarks to confirm that it still meets quality bars. And you can plug in your own use case specific test sets to check the things that actually matter for your specific application. 

## Evaluation Lab Workflow

Now, let's move to the notebook where we'll use both tools,

GuideLLM for performance, and lm_eval for accuracy, to get a complete picture of our deployment. So here again in the environment, we've already pre-warmed the vLLM server with the same model, which is Qwen3.6 billion parameters. And to start off, we're going

to do a quick test on localhost just to make sure that everything is working. and we can query that v1/models endpoint to see that we've got the model available for us to use. Let's test out the model with a quick completions request. So we'll say, hey, what is model quantization in one sentence?

and let's run this here. And you'll notice it'll probably take a few seconds for this response to come back. But we see that the model gives us a nice succinct response, but how do we actually measure the performance of this model? And that is where GuideLLM comes into play. Before we run the benchmark, we're just going to create a folder

to be able to save this data once we do the run. Now, here's the command that you'll use to run GuideLLM yourself. You can run it from the terminal, but here we're going to be running the code cell using the subprocess module. Let's break down the flags that we're passing. So first off, we're going to be targeting that localhost port 8000.

That'll point GuideLLM at our local vLLM server and hit those OpenAI compatible endpoints. We also have the profile of synchronous. It sends requests one at a time, waiting for each to finish before sending the next. And this gives us a clean baseline of single request latency,

with no batching or queuing in the picture. Other profiles like concurrent, throughput, or sweep ramp up load to stress test how the server holds up under traffic. We'll have the max amount of requests here be 10 in this situation,

which is deliberately tiny so that the cell finishes quickly in this environment. A real benchmark would use 100 or a few thousand requests, or switch to a time-bounded run with max seconds instead of max requests. For the data that we'll be

sending to the model, we'll have a total of 32 prompt tokens going in, 16 output tokens being generated, and 32 different samples of pre-generated prompts. And keeping the samples equal to or bigger than the maximum amount of requests

means no prompt repeats. So we don't accidentally inflate prefix cache hits. And finally, we'll save to that outputs directory that we created earlier. Now, let's run both of these cells and it'll take a moment to finish. When it's done, you'll have a structured benchmark file with per-request distributions ready to interpret.

The benchmark just finished up and here we can see a little bit of information about the requests that were sent to the model. And it's a bit truncated here, but you can see the total tokens per second. So we've got about 43 tokens per second,

but also the input tokens per second and output tokens per second. And this is before we actually see the inter-token latency or time for the total request to come back, which we'll see here. in just a second. But what's nice is that GuideLLM saves these results in two formats,

both a JSON and a CSV with pre-computed statistics for the means, the percentiles, the min and max and more. So you don't have to calculate all of this yourself. Let's now extract from the JSON files the key numbers that we need like the time to first token, inter-token latency,

and the end-to-end latency. This cell is going to go into our JSON file and load this so that we can extract the metrics and other information like the total request and if things were successful as well. We're going to then display this back

out and I'll run this cell right now. So we have the mean, the 50th, the 95th and the 99th percentiles for each metric. And this is what you would put in a report to your team to evaluate whether an LLM deployment meets your SLOs.

But what does this actually mean? Since averages hide outliers, you might notice that the p95, which is that 95% of requests being faster than the value is worse than the mean. And the p99 is worse still. This matters because 5 in 100 users

could be waiting many times longer than usual. When evaluating a deployment, always look at the p95 and the p99. And if there's a big gap between the mean and the p95, you've got tail latency problems and users are going to feel them. Whether it's for the time to first token, the inter-token latency or the end-to-end output metrics here

that GuideLLM shows us in this benchmark. But this performance benchmark tells you how fast the deployment is, but not whether the model gives good answers. That's a different type of test. You could have the fastest inference server in the world, but if the model's accuracy drops 15% from quantization,

it's not deployable. And that's where lm_eval comes in. lm_eval is a standardized evaluation harness that measures task performance, or how well a model answers on knowledge, reasoning, and coding benchmarks, just to name a few. While GuideLLM asks how well does this deployment perform? lm_eval will help us answer how well does this model answer.

And we'll point lm_eval at the same running server using the OpenAI completions endpoint, just like earlier and we're going to be running here hellaswag, which is a common sense reasoning benchmark that appears on most model cards. We're going to be using lm_eval's simple_evaluate function

on 20 examples to keep it quite simple, because in production, you'd probably run the full set if you had more time and more resources. lm_eval ships with hundreds of built-in tasks like MMLU, ARC, GSM8K, TruthfulQA and more. But you can also define your own

custom task by writing a small YAML config. And this is useful when you need to evaluate on your own domain specific data. But here, we'll just be calling the model and the completions endpoint with the specific tasks for 20 examples. And let's go ahead and run this cell.

It'll take a moment to finish. And once this is done, we'll go ahead and print out the accuracy metrics. So, the accuracy is going to come out for this example at about 30%. And that's with a standard error or deviation of about 10% here.

And it's kind of noisy, but that's expected because we only ran about 20 examples, but it's a starting point for a model that's quite small enough to run on modest hardware, even on your phone. These evaluations give you numerical evidence

to make educated model deployment decisions from both deployment performance using GuideLLM, to model accuracy using lm_eval. And you can use the published model card data that's been evaluated by the publisher before you even get started. In a previous lesson, you learned how to quantize a model with LLM Compressor

using GPTQ as an algorithm. In practice, quantized model publishers include accuracy tables on their model cards so users can evaluate the tradeoff without running every benchmark themselves. Here's the accuracy table from Red Hat AI's Qwen3-0.6B quantized model card. It's a W4A16 variant of the model we've been working with.

So weights at INT 4 and activations at their base release configuration. The recovery column shows how much of the base model's accuracy the quantized version retains and most benchmarks with meaningful base scores show 93 to 100% recovery. Note that the model card here reports Hellaswag with an accuracy at the release version of about 43.04,

and with this quantized model about 41.02. So you have a 95.3% recovery rate. Now, this is higher than the 30% that we received, but that gap is expected. We only ran 20 examples in a zero shot setting. While the model card uses the full

around 10000 example test set with 10 in-context examples per prompt. But you now have three sources of evidence. You have GuideLLM, which tells you how the deployment performs with latency, throughput and consistency. lm_eval, which is how the model answers

on accuracy for different types of tasks. And you have the published Model card which you can find across providers on Hugging Face and beyond, which tell you how the model performs against many benchmarks. When deciding whether to deploy a quantized model, you need both dimensions. An optimization that doubles throughput but also drops

accuracy by 15% might not be worth it. An accurate model that can't meet latency SLOs isn't deployable either. For this W4A16 model with a 50% model size reduction for an average of 4% accuracy loss on OpenLLM v1. Well, that answer depends on your use case. But check recovery on the tasks that matter to you.

Now join me for the next and final lesson where we're going to put together all the concepts that you learned about in this course.

