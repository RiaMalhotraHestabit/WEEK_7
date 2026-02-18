import json
import numpy as np
import faiss

from src.embeddings.embedder import Embedder


CHUNKS_PATH = "src/data/chunks/chunks.json"
EMBEDDINGS_PATH = "src/data/embeddings/embeddings.npy"
INDEX_PATH = "src/data/vectorstore/index.faiss"


class QueryEngine:
    def __init__(self, top_k=3):
        self.top_k = top_k

        print("Loading chunks...")
        with open(CHUNKS_PATH, "r") as f:
            self.chunks = json.load(f)

        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)

        print("Loading embedding model...")
        self.embedder = Embedder()

    def search(self, query: str):
        print(f"\nQuery: {query}")

        query_embedding = self.embedder.embed([query])

        distances, indices = self.index.search(query_embedding, self.top_k)

        results = []
        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results


if __name__ == "__main__":
    engine = QueryEngine(top_k=3)

    while True:
        query = input("\nEnter your query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = engine.search(query)

        print("\nTop Results:\n")
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Source: {result['metadata']['source']}")
            print(f"Page: {result['metadata']['page']}")
            print(f"Chunk ID: {result['metadata']['chunk_id']}")
            print(result["text"][:500])
            print("-" * 80)
