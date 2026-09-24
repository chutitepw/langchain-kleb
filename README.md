LangChain Tutorial
==============================
Tutorial on using the LangChain API 
# Requirement

- uv
- LLM model API
- Ollama (for open model deployment) 

# Setup
Clone the repository
```
git clone https://github.com/chutitepw/langchain-kleb
cd langchain-kleb
```
Set up the environment API file `.env` 
```
OPENAI_API_KEY=<your api key>
```
For open model deployment with Ollama
```
ollama pull gemma4:e4b
ollama run gemma4:e4b
```

# Run
```
uv sync
uv run python main.py
```
