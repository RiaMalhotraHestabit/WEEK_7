import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from src.embeddings.embedder import Embedder
from src.retriever.reranker import Reranker


CHUNKS_PATH = "src/data/chunks/chunks.json"
INDEX_PATH = "src/data/vectorstore/index.faiss"


class HybridRetriever:
    def __init__(self, top_k=5):
        self.top_k = top_k

        # Load chunks
        with open(CHUNKS_PATH, "r") as f:
            self.chunks = json.load(f)

        self.texts = [chunk["text"] for chunk in self.chunks]

        # Build BM25
        tokenized = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Load FAISS
        self.index = faiss.read_index(INDEX_PATH)

        # Load embedder
        self.embedder = Embedder()

        # Reranker
        self.reranker = Reranker()

    def dense_search(self, query):
        query_vec = self.embedder.embed([query])
        _, indices = self.index.search(query_vec, self.top_k * 3)
        return indices[0]

    def keyword_search(self, query):
        scores = self.bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][: self.top_k * 3]
        return top_indices
    
    def apply_filters(self, candidates, filters=None):
        if not filters:
            return candidates

        filtered = []
        for chunk in candidates:
            metadata = chunk.get("metadata", {})

            match = True
            for key, value in filters.items():
                if metadata.get(key) != value:
                    match = False
                    break

            if match:
                filtered.append(chunk)

        return filtered


    def retrieve(self, query, filters=None):
        dense_indices = self.dense_search(query)
        keyword_indices = self.keyword_search(query)

        # Merge
        combined = list(set(dense_indices.tolist() + keyword_indices.tolist()))
        candidates = [self.chunks[i] for i in combined]

        # Apply Metadata Filters (STEP 3)
        candidates = self.apply_filters(candidates, filters)

        # KEYWORD FALLBACK
        if not candidates:
            print("No results after filtering. Falling back to keyword search...")
            fallback_indices = self.keyword_search(query)
            candidates = [self.chunks[i] for i in fallback_indices]

        # Rerank
        reranked = self.reranker.rerank(query, candidates)

        # Deduplicate
        seen = set()
        final = []
        for chunk in reranked:
            text_hash = hash(chunk["text"])
            if text_hash not in seen:
                seen.add(text_hash)
                final.append(chunk)

        return final[: self.top_k]
    
# ---------------- Capstone helper ----------------
def query_hybrid_text(query, top_k=5, filters=None):
    """
    Helper function for Capstone text RAG.
    Returns {"answer": str, "confidence": float}
    """
    retriever = HybridRetriever(top_k=top_k)
    results = retriever.retrieve(query, filters)

    if not results:
        return {"answer": "No relevant documents found.", "confidence": 0.0}

    # Concatenate top chunks
    answer = " ".join([chunk["text"] for chunk in results])
    # Simple confidence heuristic: normalized number of retrieved chunks
    confidence = min(1.0, len(results)/top_k)
    return {"answer": answer, "confidence": confidence}

if __name__ == "__main__":
    retriever = HybridRetriever(top_k=5)
    query = "Explain how executive compensation works"
    results = retriever.retrieve(query)
    for i, r in enumerate(results):
        print(f"\nResult {i+1}")
        print(r["metadata"])
        print(r["text"][:300])

