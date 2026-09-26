## Setup

    mkdir gen-ai-with-llm
    cd gen-ai-with-llm
    
    uv init
    uv venv
    source .venv/bin/activate
    
    uv add openai langchain python-dotenv

Your folder will look like:

    gen-ai-with-llm/
    ├── pyproject.toml
    ├── uv.lock
    ├── .venv/
    └── main.py

Run main

    uv run python main.py