"""
Central configuration for the Healthcare RAG Assistant.
Edit these values to switch models, tune chunking, or change paths.
"""

import os

# ── LLM ───────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "mistral")   # or "llama3", "phi3", "gemma2"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 512

# ── Embeddings ────────────────────────────────────────────────────────────────
# Runs fully locally via sentence-transformers — no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Vector DB (ChromaDB) ──────────────────────────────────────────────────────
CHROMA_PERSIST_DIR  = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
CHROMA_COLLECTION   = "healthcare_docs"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 50     # overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5              # number of chunks to retrieve per query

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical information assistant. 
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say so clearly.
Do not make up medical information. Cite the document source when possible.

Context:
{context}
"""
