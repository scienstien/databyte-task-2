"""
ingestion.py
Handles document parsing, chunking, embedding, and storage into ChromaDB.

Supported file types: PDF, TXT, DOCX
"""

import os
import re
import uuid
from typing import List, Tuple

import fitz                          # PyMuPDF
import pdfplumber
import docx as python_docx
import chromadb
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
    CHUNK_SIZE, CHUNK_OVERLAP
)

# ── Singletons (loaded once) ──────────────────────────────────────────────────
_embed_model: SentenceTransformer = None
_chroma_client: chromadb.PersistentClient = None
_collection = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"[Ingestion] Loading embedding model: {EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embed_model


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF with pdfplumber fallback."""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception:
        pass

    if len(text.strip()) < 100:          # fallback for scanned/complex PDFs
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
        except Exception:
            pass

    return text


def _parse_docx(file_path: str) -> str:
    doc = python_docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _parse_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_document(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    elif ext == ".txt":
        return _parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: pdf, docx, txt")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse excess blank lines
    text = re.sub(r" {2,}", " ", text)             # collapse multiple spaces
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping character-based chunks.
    Tries to break at sentence boundaries when possible.
    """
    text = _clean_text(text)
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try to snap to the nearest sentence end before `end`
            snap = text.rfind(". ", start, end)
            if snap != -1 and snap > start + chunk_size // 2:
                end = snap + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_document(file_path: str, filename: str) -> int:
    """
    Full pipeline: parse → chunk → embed → store.
    Returns the number of chunks stored.
    """
    print(f"[Ingestion] Parsing: {filename}")
    raw_text = parse_document(file_path)

    if not raw_text.strip():
        raise ValueError("Could not extract any text from the document.")

    chunks = chunk_text(raw_text)
    print(f"[Ingestion] {len(chunks)} chunks created from '{filename}'")

    model = _get_embed_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    collection = _get_collection()

    ids       = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": filename, "chunk_index": i} for i, _ in enumerate(chunks)]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    print(f"[Ingestion] Stored {len(chunks)} chunks for '{filename}'")
    return len(chunks)


def get_collection_stats() -> Tuple[int, List[str]]:
    """Returns (total_chunk_count, list_of_unique_source_filenames)."""
    collection = _get_collection()
    total = collection.count()

    if total == 0:
        return 0, []

    results = collection.get(include=["metadatas"])
    sources = list({m["source"] for m in results["metadatas"]})
    return total, sources
