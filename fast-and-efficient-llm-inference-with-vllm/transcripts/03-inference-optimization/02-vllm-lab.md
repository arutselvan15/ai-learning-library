# vLLM Lab

> Converted transcript. Timestamp markers removed for readability.

## Lesson Transcript

In the last lesson you learned how vLLM makes LLM serving efficient through continuous batching, paged attention, and prefix caching. Now it's time to see them in action. You'll connect to a vLLM inference server, send it requests through the OpenAI compatible API, and watch these optimizations working live with the metrics.

Let's code. 

## Start a vLLM Server

To serve your model with vLLM, you need to run this command in the terminal. vllm serve with our Qwen3-0.6B model. Now, in this learning environment, we've already pre-warmed a vLLM server for you to use with this specific model, but it's already running and we'll use

the code cells in this notebook to interact with it. But before that, let's go through what each piece in this command means. So, the beginning, vllm serve. This launches vLLM's built-in inference server and it loads the model's weights into the engine with

PagedAttention, Continuous Batching, and Prefix Caching enabled by default and will expose it over HTTP on port 8000. Now, the model itself. So, this is the model's identifier on the Hugging Face Hub. On first run, vLLM will download the weights, the tokenizer, and the model's configuration

from Hugging Face onto the local cache and load that into memory. When we run the model in the future, we'll be able to reuse those cached files. This argument here loads the weights in bfloat16 precision, and the last argument we have here caps the context window

with prompt and generation at 4096 tokens. vLLM uses this to size the KV cache block pool up front. So setting it sensibly on the workloads we expect to have avoids reserving memory that you'll never use. 

## OpenAI-Compatible API

vLLM also wraps the model in an OpenAI compatible HTTP API. It implements the same routes that the OpenAI SDK calls.

So v1/models, v1/completions, and v1/embeddings. So, all of these same APIs that we use can also be replicated locally when you're running vLLM. So, let's go ahead and check to see if it's running. The VLLM_URL is going to be the address of the local server. So on port 8000. And what we're doing here

is we're sending a get request to this endpoint for the /v1/models to check and see if the server has finished loading those weights. So we'll check every about 5 seconds. here until we get a 200 response. Now, from that response, it's going to load in the specific

data that presents the model, which is going to be printed back out to make sure that we're connected to our model running in that vLLM server. Awesome. So it looks like our model is running. Now, let's send our first request.

Remember that vLLM exposes that OpenAI compatible API and we can use the standard OpenAI Python client, point that at the localhost URL at the v1 endpoint and use it exactly as if we were calling OpenAI's hosted models. So, the same client code, the same request format,

just a different base_url, which is what makes it really easy to prototype against a hosted model and then swap to a self-hosted one without rewriting your application. Now, let's ask the model if it knows about PagedAttention. And we're going to set some sampling parameters like the maximum tokens and the temperature.

And you'll also notice that Qwen 3 is a thinking model. So, to save some time, we're just going to turn that off to keep responses short and enable it later for you to see. Let's go ahead and send that request to the model. And just like that, we've got

a response back with approximately 30 tokens. And we can see that it does know what PagedAttention is. 

## Log Probabilities

Beyond just getting answers, running vLLM ourselves lets us look inside the model's decision making. For example, let's talk about what logprobs are. Because if we're asking a question to

a model like The capital of France is with some parameters like temperature and maximum tokens, well, we're going to get back an answer by default from the API. But if we go a little bit further and from the response that we get, look at the choices it had

and the log probability of those different responses, we can get a lot of additional information and understand if the model is confident in the answers that it gave. So, with this response here, if we come back down to the capital of France is,

we can see that the model had a 92.5% confidence score when selecting the answer Paris. So, this is really useful for understanding when a model is sure versus when it's guessing. 

## Server Metrics

Now, let's take a look at what's happening inside of vLLM. It exposes a metrics endpoint that we have right here

/metrics on the base URL, which is localhost for that server, and we can scrape this in order to see how many requests are running or waiting, in addition to the KV cache being used and cumulative token counts.



## Concurrent Requests

So, we're going to kind of scrape the information from there and pull these different Prometheus compatible data points and list those out here as a result. So here you can see the current metrics here. So we've got the number of requests that are running, so how many requests are active versus queued.

The GPU cache usage percentage or the KV cache memory pressure, and the prompt tokens total or generation tokens total, which is the cumulative token counts that have gone through vLLM. So, this is where things get fun. What we're going to do here is we're

going to send five requests at the same time and watch vLLM handle them. And we'll also be sending them off concurrently and while they're in flight, we're going to use that vLLM metrics function that we just showed in order to capture how many requests are running

versus how many requests are waiting to be processed. So, let's go ahead and run in this cell. As you can see, there are five concurrent requests that are being processed by vLLM in this continuous batching function. And the scheduler handles all of

these requests going at the same time. Now that we've gotten our results back, one thing that's really interesting is the total time because it's faster than running them one by one because the scheduler and continuous batching is managing these requests effectively. PagedAttention is what makes this work at scale.

The KV cache generated is divided into these fixed-size blocks that can go anywhere in memory. And when a request is done, its blocks are immediately available for use and space isn't wasted. 

## Prefix Cache Experiment

Let's see Prefix Caching because many applications use the same system prompt across every request. And without prefix caching, vLLM would have to

recompute the KV cache for that shared prefix every time. So here we're going to set up a system prompt and send five different questions with that same system prompt. The first request is going to have to pay that full prefill cost, but after that, vLLM will recognize the shared prefix and skip recomputing it.

We're going to add some extra code in order to track the prefix cache queries from the metrics endpoint before and after. And what you're going to see is that this is going to increase after each request, confirming that vLLM is

checking and reusing the cache. With these short prompts, time savings aren't very visible, but in production with thousands of instructions or few-shot examples, this eliminates a huge amount of compute. We'll have one more cell here just to print this out to our terminal.

And once we're ready, we're going to run this. Because of prefix caching, you can see it start from 235 cached queries, to 550. So that every time we sent that request with the same system prompt, we didn't have to recalculate the KV cache for those requests.

So, let's recap. You connected to a running vLLM server's OpenAI client and explored log probability as well as other optimizations like seeing PagedAttention in action and prefix caching, as well as using the metrics endpoint to watch all of these in action. In the next lesson, you'll benchmark the model with GuideLLM

and evaluate model quality with LM-Eval.

