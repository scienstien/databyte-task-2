"""
retriever.py
Embeds the user query and retrieves the top-k most relevant chunks from ChromaDB.
"""

import os
import sys
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION, TOP_K
)

# ── Singletons ────────────────────────────────────────────────────────────────
_embed_model: SentenceTransformer = None
_collection = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Embed the query, search ChromaDB, and return a list of result dicts:
        [{ "text": str, "source": str, "chunk_index": int, "distance": float }]
    """
    model = _get_embed_model()
    query_embedding = model.encode([query]).tolist()

    collection = _get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text":        doc,
            "source":      meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance":    round(dist, 4)
        })

    return retrieved
