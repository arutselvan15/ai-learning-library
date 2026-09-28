# Setup for Standalone Python Examples

[Course contents](README.md) | [Labs](chapters/05-lab-prompting.md)

The repository contains small API examples outside the three course notebooks. For those examples, install [uv](https://docs.astral.sh/uv/), then from the repository root create an environment and add the dependencies they use:

```sh
uv venv
source .venv/bin/activate
uv add openai langchain python-dotenv
uv run python main.py
```

For the complete notebooks and project scripts, install the repository dependencies from the root directory:

```sh
python -m pip install -r requirements.txt
```

The course labs install additional packages and expect specific kernels and hardware. Follow the setup cells in [Lab 1](../projects/dialogue-summarization/Lab_1_summarize_dialogue.ipynb), [Lab 2](../projects/fine-tuning/Lab_2_fine_tune_generative_ai_model.ipynb), and [Lab 3](../projects/rlhf/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb) instead of assuming this environment runs them. API example scripts may require provider credentials; never commit keys or put them in a notebook.