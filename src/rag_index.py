from typing import List, Dict, Any

import chromadb

from src.lore_loader import load_and_chunk_lore
from src.embedings import embed_texts, embed_query

_client = None
_collection = None

def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.Client()
    return _client

def build_lore_index(
    lore_dir: str = "data/lore",
    collection_name: str = "lore_collection"
):
    """
    Builds an in-memory ChromaDB index from lore text files.
    """
    global _collection

    client = get_chroma_client()

    # Load + chunk lore
    chunks = load_and_chunk_lore(lore_dir)
    print(f"[build_lore_index] Loaded {len(chunks)} chunks from {lore_dir}")

    # Embed chunks
    embeddings = embed_texts(chunks)
    print(f"[build_lore_index] Computed embeddings with shape: {embeddings.shape}")

    # (Re)create collection
    # If a collection with this name already exists, you can delete or get it
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # ignore if it doesn't exist
        pass

    collection = client.create_collection(name=collection_name)

    # Prepare IDs
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Add to Chroma
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
    )

    _collection = collection
    print(f"[build_lore_index] Collection '{collection_name}' built with {len(chunks)} docs.")

    return collection

def get_lore_collection():
    """
    Returns the current lore collection.
    Make sure build_lore_index() has been called first.
    """
    global _collection
    if _collection is None:
        # As a fallback, build from default lore dir
        _collection = build_lore_index()
    return _collection


def retrieve_lore(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Given a text query, returns top_k relevant lore chunks from the collection.
    """
    collection = get_lore_collection()

    # Embed query
    q_emb = embed_query(query)

    # Query Chroma
    result = collection.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
    )

    # result is a dict with keys: 'ids', 'documents', 'distances', 'metadatas'
    docs = result.get("documents", [[]])[0]
    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]

    out = []
    for doc_id, doc_text, dist in zip(ids, docs, distances):
        out.append(
            {
                "id": doc_id,
                "text": doc_text,
                "distance": dist,
            }
        )

    return out
