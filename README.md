LangChain Tutorial
==============================
Tutorial on using the LangChain API and Pinecone for RAG system.
# Requirement

- uv
- LLM model API
- Ollama (for open model deployment)
- Pinecone API
  
# Pinecone Setup
1. Make a Pinecone account https://www.pinecone.io/
2. Generate Pinecone API key.
3. Make a new Pinecone Index
```
Index name: medium-blogs-embeddings-index
Model: llama-text-embed-v2
Dimension: 768
```

# Local Setup
Clone the repository
```
git clone https://github.com/chutitepw/langchain-kleb
cd langchain-kleb
```
Set up the environment API file `.env` 
```
OPENAI_API_KEY=<your api key>
INDEX_NAME=<your pinecone index name>
PINECONE_API_KEY=<your pinecone api key>
```
For open model deployment with Ollama
```
ollama pull gemma4:e4b
ollama run gemma4:e4b
ollama pull nomic-embed-text
```

# Run
```
uv sync
uv run python ingest.py
uv run python main.py
```
