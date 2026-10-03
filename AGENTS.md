# AGENTS.md

Agent instructions for this learning library. Human readers should start with
[README.md](README.md). This file follows the [AGENTS.md](https://agents.md/)
format: project context, commands, and conventions for coding agents.

## Project overview

This repository stores course notes, transcripts, labs, and small Python
examples. It is a learning catalog, not a deployed application. Keep new
material readable as a course: a phase overview, ordered chapters, and a
checkpoint the learner can complete.

Courses:

| Course | Path | What it contains |
| --- | --- | --- |
| Generative AI with Large Language Models | `generative-ai-with-large-language-models/` | Phase and chapter notes, transcripts, three labs, and provider examples |
| Fast and Efficient LLM Inference with vLLM | `fast-and-efficient-llm-inference-with-vllm/` | Inference, compression, vLLM, and evaluation notes plus lab notebooks |
| Open-Source Models with Hugging Face | `open-source-models-hugging-face/` | Hands-on Hub model notebooks. See its [README](open-source-models-hugging-face/README.md) for the lab and model index |

The root README is the course catalog. When a course is added or renamed,
update that catalog and the course's own README in the same change. The
closest `AGENTS.md` to an edited file wins; add a nested file only when a
course needs different commands.

## Setup commands

Python 3.11 is pinned in `.python-version`.

```bash
uv venv
source .venv/bin/activate
uv sync
```

For the documented notebook and script dependencies:

```bash
python -m pip install -r requirements.txt
```

Provider examples under
`generative-ai-with-large-language-models/examples/` load keys from the
environment. Copy `.env-example` to `.env` locally and fill `GOOGLE_API_KEY`
and `OPENAI_API_KEY` there. Hugging Face access, when a gated model or private
Space requires it, uses `HF_TOKEN`.

```bash
uv run python main.py
```

Run Jupyter from the course directory whose notebooks you are executing.
Hugging Face labs resolve models from `open-source-models-hugging-face/models/`.

```bash
hf download facebook/blenderbot-400M-distill \
  --local-dir open-source-models-hugging-face/models/facebook/blenderbot-400M-distill
```

## Repository map

- `generative-ai-with-large-language-models/phases/NN-name/MM-chapter/`: chapter README, optional `transcripts/`, and a `lab/` when the chapter is an exercise.
- `generative-ai-with-large-language-models/setup.md`: environment notes for the standalone examples. Course notebooks have their own setup cells.
- `fast-and-efficient-llm-inference-with-vllm/phases/`: the same phase and chapter pattern. Lab notebooks live beside the chapter README.
- `open-source-models-hugging-face/phases/`: numbered notebooks. There is no phase `06`; keep the existing filenames.
- `requirements.txt` and `pyproject.toml`: shared Python dependencies. Change both when a root dependency changes.
- `.env-example`: key names only. `.env`, `.venv`, and `.idea` stay untracked.

## Working on course material

- Read the phase README before editing a chapter. Preserve the previous, contents, and next links.
- Chapter pages lead with the goal, the material to use, the order of work, and a checkpoint. Match that shape when adding a chapter.
- Transcripts are source material. Summarize them into the chapter page; leave the transcript text in place unless the task is to correct that transcript.
- Link labs, datasets, and models with relative paths and stable URLs. Name the model ID used by the lab, such as `google/flan-t5-base` or `facebook/blenderbot-400M-distill`.
- Hugging Face notebooks load most weights from `./models/<org>/<model>`. Labs 03 and 05 load Hub IDs directly. Keep those two loading styles unless the notebook itself is being changed.
- Edit notebook source cells. Leave saved outputs and widget state unchanged unless the task is to refresh a run.
- A new course needs a root README entry, its own README with outcome and map, and ordered phase folders.

## Code style

- Python 3.11. Small teaching scripts use top-level imports and a direct `main` path where the file is an entry point.
- Use 4-space indentation. Keep example scripts short enough to read beside the chapter that points to them.
- Load provider clients from environment variables through `python-dotenv` or `os.getenv`. Follow the pattern in `generative-ai-with-large-language-models/examples/open-ai.py`.
- Pin teaching dependencies when a lab depends on a known library behavior. The current pins include `transformers==4.38.2`, `torch==2.5.1`, `datasets==2.17.0`, and `peft==0.3.0`.
- Prefer the existing course vocabulary: phase, chapter, lab, checkpoint, and model ID.

## Testing instructions

There is no unit-test suite. Before finishing a change, run the checks that match the files you touched:

```bash
python -m compileall -q main.py generative-ai-with-large-language-models
python -m json.tool open-source-models-hugging-face/phases/01_chatbot_conversation.ipynb >/dev/null
```

Repeat the JSON check for every notebook you edit. For a Markdown change, open the edited page and confirm relative links resolve to files in the tree.

Do not download model weights, start vLLM, launch Gradio, or run fine-tuning, RLHF, or long notebook cells unless the user asks for that run. Those labs need extra disk, and several expect a GPU or the original course instance. When a run is requested, use the model named in that lab and record the command and result in the chapter notes.

## Security

- Keep secrets in `.env` or the process environment. Never write API keys, Hugging Face tokens, private keys, or connection strings into notebooks, scripts, or Markdown.
- `.env-example` contains empty key names. Do not replace those with real values.
- Do not commit `models/`, Hugging Face cache directories, checkpoints, or datasets downloaded during a lab.
- Public Hub models are the default. If a lab needs a gated model, document the environment variable and leave the value unset.
- Provider examples call external APIs. Do not add a new provider call that sends private notebook data unless the chapter is explicitly about that API.

## PR instructions

- Commit subjects follow the existing style: `docs: ...` for notes and labs, `chore: ...` for structure and dependency maintenance.
- Title a pull request with the course or area first, then the change, for example `[huggingface] Add lab model index`.
- Include the course path, the chapters or notebooks changed, and the checks that were run.
- Leave unrelated course rewrites and notebook output refreshes out of the change.
