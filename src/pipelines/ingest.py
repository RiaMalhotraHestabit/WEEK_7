import os
import json
import numpy as np

from src.utils.document_loader import load_documents_from_folder
from src.utils.text_cleaner import clean_text
from src.utils.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_index import build_faiss_index


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


if __name__ == "__main__":
    main()
