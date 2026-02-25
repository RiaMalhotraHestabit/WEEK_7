import faiss
import os

def build_faiss_index(embeddings, save_path):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    faiss.write_index(index, save_path)

    return index

def load_faiss_index(index_path):
    return faiss.read_index(index_path)

def search_faiss_index(index, query_vector, top_k=5):
    distances, indices = index.search(query_vector, top_k)
    return distances, indices