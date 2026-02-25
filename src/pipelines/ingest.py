import os
import json
import numpy as np

from src.utils.document_loader import load_documents_from_folder
from src.utils.text_cleaner import clean_text
from src.utils.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_index import build_faiss_index, load_faiss_index, search_faiss_index

RAW_DATA_PATH = "src/data/raw"
CHUNK_SAVE_PATH = "src/data/chunks/chunks.json"
EMBED_SAVE_PATH = "src/data/embeddings/embeddings.npy"
INDEX_SAVE_PATH = "src/data/vectorstore/index.faiss"


def main():
    print("Loading documents...")
    documents = load_documents_from_folder(RAW_DATA_PATH)
    print(f"Loaded {len(documents)} pages.")

    print("Cleaning & chunking...")
    all_chunks = []

    for doc in documents:
        doc["text"] = clean_text(doc["text"])
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks.")

    os.makedirs("src/data/chunks", exist_ok=True)
    with open(CHUNK_SAVE_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print("Generating embeddings...")
    embedder = Embedder()
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedder.embed(texts)

    os.makedirs("src/data/embeddings", exist_ok=True)
    np.save(EMBED_SAVE_PATH, embeddings)

    print("Building FAISS index...")
    build_faiss_index(embeddings, INDEX_SAVE_PATH)

    print("✔ Documents loaded")
    print("✔ Chunks created")
    print("✔ Embeddings generated")
    print("✔ Vector DB initialized")


# ---------------- Capstone Query Function ----------------
def query_text(query, top_k=5):
    """
    Query the FAISS index with a text string.
    Returns: {"answer": str, "confidence": float}
    """
    # Load FAISS index
    index = load_faiss_index(INDEX_SAVE_PATH)
    embeddings = np.load(EMBED_SAVE_PATH)
    embedder = Embedder()

    # Embed query
    query_embedding = embedder.embed([query])[0]

    # Retrieve top_k chunks
    scores, indices = search_faiss_index(index, query_embedding, top_k=top_k)

    # Load chunks
    with open(CHUNK_SAVE_PATH, "r") as f:
        chunks = json.load(f)

    retrieved_texts = [chunks[i]["text"] for i in indices]

    # Simple answer generation: concatenate top chunks (replace with LLM if needed)
    answer = " ".join(retrieved_texts)
    confidence = float(np.mean(scores))  # simple heuristic

    return {"answer": answer, "confidence": confidence}


if __name__ == "__main__":
    main()