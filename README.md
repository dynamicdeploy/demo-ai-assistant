# Streamlit AI assistant &mdash; a documentation chatbot for Streamlit

Ever wanted to chat with the Streamlit documentation? Well, with the power of
LangChain and modern LLMs (OpenAI, Anthropic Claude, or Ollama), now you can!

## Running it yourself

### Prerequisites

1. Get the code:

   ```sh
   $ git clone https://github.com/streamlit/demo-ai-assistant
   ```

2. Start a virtual environment and get the dependencies (requires uv):

   ```sh
   $ uv venv
   $ .venv/bin/activate
   $ uv sync
   ```

   Or using pip:

   ```sh
   $ python -m venv venv
   $ source venv/bin/activate  # On Windows: venv\Scripts\activate
   $ pip install -e .
   ```

### Configuration

Create a `.env` file in the project root with at least one of the following LLM provider configurations:

**Option 1: OpenAI**
```env
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o
```

**Option 2: Anthropic Claude**
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
MODEL_NAME=claude-3-5-sonnet-20241022
```

**Option 3: Ollama (local)**
```env
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama3
```

**Note:** 
- `MODEL_NAME` is optional and will use provider defaults if not set
- Priority: OpenAI > Anthropic > Ollama (first available will be used)
- See `.env.example` for a template

### Start the app

```sh
$ streamlit run streamlit_app.py
```

The app will automatically detect which LLM provider to use based on your `.env` configuration.
