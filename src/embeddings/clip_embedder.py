import torch
import open_clip
from PIL import Image

class CLIPEmbedder:
    def __init__(self, device="cpu"):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def embed_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features.cpu().numpy()[0]

    def embed_text(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        return text_features.cpu().numpy()[0]