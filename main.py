from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
# Ollama for local model deployment
from langchain_ollama import ChatOllama

# RAG data retrival
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

import os
import pandas as pd
import numpy as np

from langchain.tools import tool
from langchain.agents import create_agent

from keras.models import load_model

_model_cache = {}

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)

# Define a tool to classify a CSV file using a Keras model
@tool
def classify_csv(file_path: str, model_path: str) -> str:
    """Classify a hardware-performance-counter CSV with a Keras model. Each data column represents a different hardware event: Branch instruction retired,  Branch misses,  L2 cache references,  L2 cache misses, and  Instruction retired..
    Args:
        file_path: Path to the CSV file to classify.
        model_path: Path to the .keras model file.
    Returns:
        A summary of predicted classes (0=benign, 1=ransomware, 2=spectre).
    """

    # Load the model from cache or disk
    if model_path not in _model_cache:
        _model_cache[model_path] = load_model(model_path)
    model = _model_cache[model_path]

    # Load data sample
    data = pd.read_csv(file_path, nrows=500).to_numpy()
    window = 50
    X = np.array([data[i:i + window] for i in range(len(data) - window)])
    print(f"Data shape: {X.shape}")

    # Predict classes
    preds = model.predict(X, verbose=0).argmax(axis=-1)
    labels = {0: "benign", 1: "ransomware", 2: "spectre"}
    counts = {labels.get(int(k), str(k)): int(v)
              for k, v in zip(*np.unique(preds, return_counts=True))}
    majority = max(counts, key=counts.get)
    print(f"Windows classified: {len(preds)}. Counts: {counts}. Majority: {majority}.")

    return f"Windows classified: {len(preds)}. Counts: {counts}. Majority: {majority}."

@tool
def search_manual(query: str) -> str:
    """Search the reference manual for background on hardware performance counters,
    ransomware and Spectre behavior, and how to interpret classification results.
    Args:
        query: A natural-language question or keywords to look up.
    Returns:
        The most relevant manual excerpts with their sources.
    """
    results = vector_store.similarity_search(query, k=4)
    return "\n\n---\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', '?'))}"
        f" p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in results
    )

tools = [classify_csv, search_manual]

def main():

    test_data_path = "data/test/test-sodinokibi.csv"
    model_path = "model/cnn-amd-all.keras"

    # For Ollama local model deployment
    llm = ChatOllama(temperature=0, model="gemma4:e4b")

    # Create an agent with the model and tools
    agent = create_agent(
        model=llm, 
        tools=tools,
        system_prompt=(
            "You are a security data scientist. First use classify_csv to classify the dataset. "
            "Then use search_manual to look up what the predicted class and the relevant "
            "performance counters mean. Base your explanation on the manual excerpts and "
            "cite their sources. If the manual doesn't cover something, say so."
        ),
    )

    # Invoke agent to get response
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Classify the dataset at {test_data_path} "
                       f"using the model at {model_path}.",
        }]
    })
    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()
