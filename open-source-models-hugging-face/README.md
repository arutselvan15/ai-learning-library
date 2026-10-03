# Open-Source Models with Hugging Face

Hands-on notebooks for learning how to use open-source models from the
[Hugging Face Hub](https://huggingface.co/models). The labs progress from
language tasks to audio, computer vision, multimodal applications, Gradio
interfaces, and deployment to Hugging Face Spaces.

## Learning path and model quick reference

| Lab | Topic | Model(s) used | Purpose |
| --- | --- | --- | --- |
| [01](phases/01_chatbot_conversation.ipynb) | Chatbot conversation | [`facebook/blenderbot-400M-distill`](https://huggingface.co/facebook/blenderbot-400M-distill) | Generate conversational responses with a compact BlenderBot model. |
| [02](phases/02_translation_and_summarization.ipynb) | Translation and summarization | [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M), [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn) | Translate text across languages with NLLB and summarize long English text with BART. |
| [03](phases/03_sentence_embeddings.ipynb) | Sentence embeddings | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Convert sentences into dense vectors for semantic similarity comparisons. |
| [04](phases/04_zero-shot_audio_classification.ipynb) | Zero-shot audio classification | [`laion/clap-htsat-unfused`](https://huggingface.co/laion/clap-htsat-unfused) | Classify sounds against user-provided text labels without task-specific training. |
| [05](phases/05_automatic_speech_recognition.ipynb) | Automatic speech recognition | [`distil-whisper/distil-small.en`](https://huggingface.co/distil-whisper/distil-small.en) | Transcribe English speech efficiently and expose the workflow through Gradio. |
| [07](phases/07_text_to_speech.ipynb) | Text to speech | [`kakao-enterprise/vits-ljs`](https://huggingface.co/kakao-enterprise/vits-ljs) | Synthesize English speech from text with VITS. |
| [08](phases/08_object_detection.ipynb) | Object detection and audio narration | [`facebook/detr-resnet-50`](https://huggingface.co/facebook/detr-resnet-50), [`kakao-enterprise/vits-ljs`](https://huggingface.co/kakao-enterprise/vits-ljs) | Detect and locate objects with DETR, then narrate the detections with speech synthesis. |
| [09](phases/09_segmentation.ipynb) | Segmentation and depth estimation | [`Zigeng/SlimSAM-uniform-77`](https://huggingface.co/Zigeng/SlimSAM-uniform-77), [`Intel/dpt-hybrid-midas`](https://huggingface.co/Intel/dpt-hybrid-midas) | Generate or prompt image masks with SlimSAM and estimate scene depth with DPT. |
| [10](phases/10_image_retrieval.ipynb) | Image retrieval | [`Salesforce/blip-itm-base-coco`](https://huggingface.co/Salesforce/blip-itm-base-coco) | Score image-text matches to retrieve images relevant to a text query. |
| [11](phases/11_image_captioning.ipynb) | Image captioning | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) | Generate conditional and unconditional natural-language descriptions of images. |
| [12](phases/12_visual_q_and_a.ipynb) | Visual question answering | [`Salesforce/blip-vqa-base`](https://huggingface.co/Salesforce/blip-vqa-base) | Answer natural-language questions about the content of an image. |
| [13](phases/13_zero_shot_image_classification.ipynb) | Zero-shot image classification | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | Classify images using arbitrary candidate labels through image-text similarity. |
| [14](phases/14_deployment.ipynb) | Gradio and Hugging Face Spaces | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) | Package the image-captioning model in a Gradio app and deploy it to a Space. |

> The repository currently has no phase `06` notebook; numbering follows the
> existing filenames.

## What you will learn

- Build inference pipelines with `transformers`.
- Create sentence embeddings with `sentence-transformers`.
- Work with text, audio, images, and multimodal inputs.
- Load models from the Hub or from a local model cache.
- Build interactive demos with Gradio.
- Deploy a model-backed application to Hugging Face Spaces.

## Getting started

### Prerequisites

- Python 3.10 or later
- JupyterLab or VS Code with Jupyter support
- Enough disk space for the selected model weights
- Internet access for the initial model and dataset downloads

A GPU is helpful for the larger vision and audio models, but many labs can run
on a CPU with slower inference.

### Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip jupyter
```

Each notebook contains its own package installation cell. Common dependencies
include `transformers`, `torch`, `sentence-transformers`, `datasets`, `gradio`,
`soundfile`, `librosa`, `timm`, and `torchvision`.

### Model locations

Most notebooks expect downloaded model repositories under `./models`, for
example:

```text
models/
├── facebook/blenderbot-400M-distill/
├── facebook/nllb-200-distilled-600M/
├── Salesforce/blip-image-captioning-base/
└── openai/clip-vit-large-patch14/
```

Run Jupyter from this directory so those relative paths resolve correctly:

```bash
jupyter lab
```

Models can be downloaded with the current Hugging Face CLI:

```bash
hf download facebook/blenderbot-400M-distill \
  --local-dir ./models/facebook/blenderbot-400M-distill
```

Repeat with the model ID required by the lab. Labs 03 and 05 currently load
their models directly by Hub ID, so the Hugging Face libraries manage their
local cache automatically.

## Suggested order

1. Start with labs 01-03 for text pipelines and embeddings.
2. Continue with labs 04, 05, and 07 for audio understanding and generation.
3. Complete labs 08-13 for vision and multimodal tasks.
4. Finish with lab 14 to package and deploy a Gradio application.

## NLP and generative AI

**Natural Language Processing (NLP)** focuses on understanding and processing
human language. Typical NLP tasks include classification, translation,
information extraction, and speech recognition.

**Generative AI** creates new content such as text, images, audio, video, or
code. The two areas overlap: a language model can use NLP techniques to
understand a prompt and then generate a response.

- NLP example: determine whether a customer review is positive or negative.
- Generative AI example: write a response to that customer review.

## Authentication and secrets

Public models usually do not require authentication. If a gated model or
private Space needs a Hugging Face token, keep it outside the notebook:

```bash
export HF_TOKEN="your-token"
hf auth whoami
```

Do not commit tokens or `.env` files. Add `.env` to `.gitignore` when using
`python-dotenv`.
