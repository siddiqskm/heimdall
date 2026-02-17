# heimdall/core/embedder.py

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, text: str):
        return self.model.encode(
            [text.lower().strip()],
            normalize_embeddings=True
        )[0]
