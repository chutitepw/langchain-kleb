from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag import DOCS_DIR, get_vector_store

# Load every manual in docs/ (one Document per PDF page so citations can show page numbers)
docs = []
for path in Path(DOCS_DIR).rglob("*"):
    if path.suffix.lower() == ".pdf":
        for page_num, page in enumerate(PdfReader(path).pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(text, metadata={"source": str(path), "page": page_num}))
    elif path.suffix.lower() in {".md", ".txt"}:
        docs.append(Document(path.read_text(encoding="utf-8"), metadata={"source": str(path)}))

if not docs:
    raise SystemExit(f"No .pdf, .md or .txt files found in {DOCS_DIR}/")
print(f"Loaded {len(docs)} pages/documents")

# Split into overlapping chunks so each embedding covers one focused idea
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# Rebuild the collection from scratch so removed or edited manuals don't leave stale chunks
vector_store = get_vector_store()
vector_store.reset_collection()
vector_store.add_documents(chunks)
print(f"Stored {len(chunks)} chunks in Chroma")
