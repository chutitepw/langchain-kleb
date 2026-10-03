from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = os.environ["INDEX_NAME"]
DOCS_DIR = Path("docs")

# Create the index if it doesn't exist (dimension must match the embedding model)
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Load every manual in docs/
docs = []
for path in DOCS_DIR.rglob("*"):
    if path.suffix.lower() == ".pdf":
        docs.extend(PyPDFLoader(str(path)).load())
    elif path.suffix.lower() in {".md", ".txt"}:
        docs.extend(TextLoader(str(path), encoding="utf-8").load())
print(f"Loaded {len(docs)} pages/documents")

# Split into overlapping chunks so each embedding covers one focused idea
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# Embed and upload; stable ids make re-running overwrite instead of duplicate
embeddings = OllamaEmbeddings(model="nomic-embed-text")
ids = [f"{Path(c.metadata['source']).name}-{c.metadata.get('page', 0)}-{i}"
       for i, c in enumerate(chunks)]
PineconeVectorStore.from_documents(
    chunks, embeddings, index_name=INDEX_NAME, ids=ids
)
print("Ingestion complete")
