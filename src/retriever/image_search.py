import sys
from pathlib import Path
import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.embeddings.clip_embedder import CLIPEmbedder

class ImageSearchEngine:
    def __init__(self, vector_store_dir="vectorstore"):
        self.embedder = CLIPEmbedder()
        self.client = chromadb.PersistentClient(path=vector_store_dir)
        self.collection = self.client.get_collection("image_rag")
        print(f"Search engine ready. Total docs: {self.collection.count()}")

    def text_to_image(self, query, top_k=5):
        print(f"\n[Text → Image] Query: '{query}'")
        embedding = self.embedder.embed_text(query)
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        print(f"Top {top_k} results:")
        for i, meta in enumerate(results["metadatas"][0], 1):
            score = 1 - results["distances"][0][i-1]
            print(f"  {i}. {meta['filename']}  (score: {score:.3f})")

    def image_to_image(self, image_path, top_k=5):
        print(f"\n[Image → Image] Query: '{Path(image_path).name}'")
        embedding = self.embedder.embed_image(image_path)
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        print(f"Top {top_k} similar images:")
        for i, meta in enumerate(results["metadatas"][0], 1):
            score = 1 - results["distances"][0][i-1]
            print(f"  {i}. {meta['filename']}  (score: {score:.3f})")

    def image_to_text(self, image_path, top_k=3):
        print(f"\n[Image → Text] Query: '{Path(image_path).name}'")
        embedding = self.embedder.embed_image(image_path)
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        context = ""
        for meta in results["metadatas"][0]:
            context += f"Caption: {meta.get('caption', '')}\n"
            context += f"OCR: {meta.get('ocr_text', '')}\n\n"

        print(f"Answer:")
        print(f"  This image is a {results['metadatas'][0][0].get('caption', '')}")
        print(f"  Related text found: {results['metadatas'][0][0].get('ocr_text', '')[:100]}")

if __name__ == "__main__":
    engine = ImageSearchEngine(vector_store_dir="vectorstore")

    # Mode 1: Text → Image (only filenames)
    engine.text_to_image("bar chart with categories", top_k=3)

    # Mode 2: Image → Image (only filenames)
    engine.image_to_image("src/data/raw/images/bar_chart_1.png", top_k=3)

    # Mode 3: Image → Text (only text answer)
    engine.image_to_text("src/data/raw/images/bar_chart_1.png", top_k=3)