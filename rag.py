from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Shared settings so ingest.py and main.py always use the same store and embeddings
DOCS_DIR = "docs"
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "kleb-manual"
EMBEDDING_MODEL = "nomic-embed-text"


def get_vector_store() -> Chroma:
    """Open the persistent local Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=PERSIST_DIR,
    )
