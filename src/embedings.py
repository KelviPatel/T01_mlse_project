from typing import List

from sentence_transformers import SentenceTransformer
import numpy as np

# Global variable to cache the model
_embedding_model = None

def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Lazily loads and returns the sentence transformer model.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Given a list of text strings, returns a 2D numpy array of embeddings.
    Shape: (len(texts), embedding_dim)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string and returns a 1D numpy array.
    """
    model = get_embedding_model()
    embedding = model.encode([query])[0]
    return np.array(embedding)
