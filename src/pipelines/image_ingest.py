import os
import uuid
from pathlib import Path
from PIL import Image
import chromadb
import sys
import pytesseract
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.embeddings.clip_embedder import CLIPEmbedder

class ImageIngestionPipeline:
    def __init__(self, vector_store_dir="vectorstore"):
        # CLIP
        self.embedder = CLIPEmbedder()

        # BLIP captioning
        print("Loading BLIP captioning model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model.eval()
        print("BLIP ready ✓")

        # ChromaDB
        self.client = chromadb.PersistentClient(path=vector_store_dir)
        self.collection = self.client.get_or_create_collection(
            name="image_rag",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Vector store ready. Total docs: {self.collection.count()}")

    def extract_ocr(self, image):
        try:
            gray = image.convert("L")
            text = pytesseract.image_to_string(gray, config="--psm 3")
            return text.strip()
        except Exception as e:
            print(f"  OCR error: {e}")
            return ""

    def generate_caption(self, image):
        try:
            inputs = self.blip_processor(image, return_tensors="pt")
            with torch.no_grad():
                output = self.blip_model.generate(**inputs, max_new_tokens=50)
            caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"  Caption error: {e}")
            return ""

    def ingest_image(self, image_path):
        image_path = Path(image_path)
        print(f"\nIngesting: {image_path.name}")

        image = Image.open(image_path).convert("RGB")

        # Step 1: OCR
        ocr_text = self.extract_ocr(image)
        print(f"  OCR: {ocr_text[:50]}..." if ocr_text else "  OCR: none")

        # Step 2: Caption
        caption = self.generate_caption(image)
        print(f"  Caption: {caption}")

        # Step 3: CLIP embedding
        embedding = self.embedder.embed_image(image_path)

        # Step 4: Store everything
        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "source": str(image_path),
                "filename": image_path.name,
                "ocr_text": ocr_text[:500],
                "caption": caption,
            }],
            documents=[f"Caption: {caption}\nOCR: {ocr_text[:200]}"]
        )
        print(f"  Stored ✓")
        return doc_id

    def ingest_directory(self, directory):
        directory = Path(directory)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        files = [f for f in directory.iterdir() if f.suffix.lower() in extensions]

        print(f"\nFound {len(files)} images\n")
        for i, file in enumerate(files, 1):
            print(f"[{i}/{len(files)}]", end=" ")
            try:
                self.ingest_image(file)
            except Exception as e:
                print(f"  Error: {e}")

        print(f"\nDone! Total in store: {self.collection.count()}")

#----for capstone project----

def query_image(file_path, top_k=5):
    """
    Capstone helper for /ask-image
    """
    pipeline = ImageIngestionPipeline(vector_store_dir="vectorstore")

    # Embed input image
    embedding = pipeline.embedder.embed_image(file_path)

    # Query ChromaDB
    results = pipeline.collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        include=['documents','metadatas','distances']
    )

    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    if not docs:
        return {"answer": "No similar images found.", "confidence": 0.0}

    # Concatenate captions + OCR
    answer_texts = []
    for meta in metadatas:
        text = f"Caption: {meta.get('caption','')}\nOCR: {meta.get('ocr_text','')}"
        answer_texts.append(text)

    answer = "\n\n".join(answer_texts)
    # Simple confidence heuristic: inverse of average distance (normalized)
    confidence = float(max(0.0, 1.0 - sum(distances)/len(distances)))

    return {"answer": answer, "confidence": confidence}

if __name__ == "__main__":
    pipeline = ImageIngestionPipeline(vector_store_dir="vectorstore")
    pipeline.ingest_directory("src/data/raw/images")