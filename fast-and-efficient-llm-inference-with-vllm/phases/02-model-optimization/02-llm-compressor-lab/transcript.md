# LLM Compressor Lab

> Converted transcript. Timestamp markers removed for readability.

## Lesson Transcript

In the previous lesson, you learned about quantization. what it is, why it matters, and how it reduces the cost of running AI models. But in this lesson, you're going to actually do it. We'll take a full precision model, compress it with a tool called LLM Compressor,

and compare the sizes before and after. We'll also measure whether the compressed model still works well. So, it's time to compress some weights. 

## Compression Workflow

Here's the steps for compressing an LLM using the LLM Compressor open source tool. There's going to be four steps. First off

is choosing a model like one from Hugging Face. The second step is picking the algorithm, as different algorithms have different tradeoffs in terms of compression speed and the retention of their accuracy. Next is choosing the Quantization Scheme, where you decide the precision for weights and activations of the model.

And finally is inference with vLLM, because once you have a compressed model, Inference engines like vLLM can load it directly, but we'll talk about that in the next lesson. 

## Model and Calibration Data

When you're choosing a model, typically you'll start from either Hugging Face, which is like the GitHub of models,

or an internal registry within your own organization. But that's not all because most quantization algorithms need representative data to calibrate the compression. They use that data to understand which weights matter most and how to minimize accuracy loss during quantization. 

## Algorithm Choices

Here's a comparison of the main algorithms available in LLM Compressor,

the open source tool that you'll be using today. And we'll dive into a few of these here in just a second. Round-to-nearest is the simplest. It's fastest to run and it gives you a good baseline to start from. AWQ or activation aware weight quantization

gives the best balance of accuracy and speed, especially on NVIDIA hardware. But GPTQ is the industry standard. Although it can require more VRAM to be needed during compression. Sparse-GPT handles sparsification, but only in unique cases using specific hardware like the NVIDIA H100.

And finally, you can reshape the weights or activations before compressing them with one of these algorithms, so that less information is lost. Smoothing flattens the spikes that the model might have, and transformations apply rotations to the weights to lose less information when you're going to quantize.



## Round-to-Nearest

The simplest approach is Round-to-nearest, which is exactly what it sounds like. Each weight is rounded to the nearest value in the target precision. So, there's no calibration data needed and it's very fast to run, but you get accuracy degradation specifically at lower bits like int4. So, it's a good baseline but not the

right choice for production. So, what approaches work better? 

## AWQ

Let's start with AWQ, which is quite popular and based off of an interesting observation that not all weights are equally important. And some weights, when you change them even slightly, cause large changes in the model's outputs.

Others can be rounded quite aggressively without much effect. And so what AWQ does is it figures out which is which by looking at the activation magnitudes during a pass of calibration data. So weights that correspond to large activations are treated more carefully,

and the rest are compressed more aggressively. AWQ is also computationally lighter than the next algorithm GPTQ. So calibration runs faster and requires less VRAM. 

## GPTQ

GPTQ takes a more mathematically rigorous approach. Its idea is that given we're going to quantize this model

and introduce some error, how do we compensate for that error in the remaining weights so that the overall output changes as little as possible. To do this, it computes the Hessian of the loss with respect to the weights. Essentially a measure of the curvature that tells you

how sensitive the model output is to changes in each specific weight. And then it works through the weights layer by layer, quantizing each one and updating the remaining weights to compensate. As computing and inverting those Hessians is quite expensive, the result is very high accuracy, often better than AWQ

on certain benchmarks, although the tradeoff is more compute and memory costs needed. As computing and inverting those Hessians is expensive. GPTQ is also the most widely supported algorithm and a safe choice for sharing a quantized model with others. 

## Lab Workflow

Now, let's move to the notebook where we're

going to use the open source LLM compressor tool to quantize a model and compare the difference before and after we do this compression. First, we'll import our standard libraries like torch, the Hugging Face transformers for loading models and tokenizers. And we're going to be working with Qwen3-0.6B as our base model.

It's small enough to work with in this environment, but it's a real capable language model. And you've got folders for both the original and quantized model weights in your environment already. Now, we need to specify a recipe which tells LLM Compressor how to quantize.

We're going to be using the GPTQ algorithm today, using calibration data to find the optimal quantized values for each weight. And then we're going to be going for four-bit weights and 16-bit activations. which can lead to a roughly 50% total reduction for this model. We're also going to be targeting the Linear

layers where the vast majority of parameters live, and ignoring the lm_head, which is the output layer that maps tokens to vocabularies so that we can keep this compressed model also precise. Now, a recipe could be a list of quantization algorithms, but for here we're using just one algorithm.

Now, to produce the quantized model, we need to import oneshot. It takes the model along with some calibration data. and a recipe together to tell us how to do compression in a single pass. That recipe is right from here. Now, the dataset parameter. Let's take a look at this

because this specifies what text to use for calibration and here we're going to be using wikitext 2, a standard benchmark of Wikipedia articles and the same dataset you'll use later on for perplexity evaluation. the max_seq_length here and the number of calibration samples

control the calibration pass that we're going to do with that dataset. So the num_calibration_samples is how many sequences are run through the model. More samples give a better picture of weight importance, but past a few hundred, the accuracy gains become tiny while runtime keeps growing.

So 256 is a solid default. Now, the max sequence length. This is the max token length per sample that we're using. So longer sequences let the quantizer see how weights behave across realistic context lengths. And samples beyond this get truncated.

These parameters all help to improve accuracy as we go from default full precision of BFloat16 to INT4 precision. of these weights. Since quantization can take several minutes and benefits from a GPU, we've already run it ahead of time and provided the quantized model in the output directory folder, which is Qwen3-0.6B W4A16.

This if statement just checks and makes sure that we skip rerunning quantization when the folder already exists so that you can move straight to evaluation. Awesome. So let's see what quantization actually saved us. This cell here goes through both model directories, the original released model and the quantized model,

and prints a comparison of the sizes of both models. Now, you might expect a 75% reduction since we went from 16-bit to 4-bit weights, but the actual reduction is 42%. And the reason for that is that the linear layer weights are the only ones that are quantized for. The rest of the model,

including the LM head and normalization layers stay at a higher precision. So, this four times compression applies to the bulk of the parameters, which are the linear layers that dominate the model. But the unquantized pieces pull the overall reduction down to 42%. This ratio improves with larger models where the linear weights make up an even bigger share of the total size,

like a 70B model quantized the same way gets much closer to the theoretical four times compression of the model. 

## Compare Generated Output

But smaller model sizes are only useful if the model still works. So, let's do a comparison of both using the same prompt. First, we're going to load in the base_model with its original weights on cpu

and give it a prompt machine learning is a branch of And we're going to generate 60 tokens with standard sampling settings. This is going to act as our baseline. Take a look at the response of this model and we'll compare this with the quantized model here in a second. Now we're using the quantized model with

the exact same prompt and generation settings. The only difference is that the model has four-bit weights instead of 16-bit. So, let's run this cell. Now, it'll probably take some time on your environment to get that response back. Here's the output, and let's compare the results between both models.

The quantized model should produce output similar to the baseline, maybe not word for word, but quite similar to the original model. That's huge compression but at a minimal quality loss, and that's the goal. 

## Perplexity Validation

Side-by-side text gives you an intuition, but we need a number, and that's where perplexity comes in.

It's a standard metric for language models that measures how well the model predicts text. So, lower is better, and if quantization has degraded the model, then the perplexity will be noticeably higher. Here we define a calculate_perplexity function that takes a chunk from the wikitext-2 dataset

and slides a window across it. So at each position, it's computing the cross-entropy loss between the model's predicted next token distribution and the actual next token. essentially how surprised the model is by the token that actually came next. And when we exponentiate the average loss,

that gives us what's known as perplexity. And each window moves forward by stride tokens and overlaps with the previous one. The sliding window with stride lets us evaluate long text without feeding it all at once, while still giving each token a reasonable amount of left context.

We load the test split. It's the same data set as the calibration data, but a held out portion so that there's no data leakage. Now, let's compute perplexity for the quantized model first, as it's already in memory. It'll take a moment as we're running

the full test data through the model. So we've got a result back for our quantized model perplexity at 35.48 Now, let's run it for our base model as well. Once the base perplexity calculation came back, now we have a 32.79 that we can use to compare between

the original model and the quantized model side by side. We'll use this simple calculation in order to calculate the difference in perplexity between the base model and that quantized model. And for us, it's about 8% But remember, the lower perplexity is better.

And the question is whether this increase is acceptable for your use case. Because for most production deployments, a few percent increase in perplexity is well worth the reduction in model size and infrastructure savings that come with it. Okay, so let's recap. You've

learned how the one shot API from LLM Compressor applies post-training quantization with a GPTQ recipe. You compared model sizes and saw the concrete reduction from W4A16 quantization. And tested both models on the same prompt to verify that the output still makes sense.

And you measured perplexity to put a number on that accuracy tradeoff. In the next lesson, we're going to switch over to inference optimizations and learn more about vLLM. We'll see you there.

