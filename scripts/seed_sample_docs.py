"""
seed_sample_docs.py
Pre-loads the sample medical documents in sample_docs/ into ChromaDB.
Run this once after setup to have data ready to query immediately.

Usage:
    python scripts/seed_sample_docs.py
"""

import os
import sys

# Ensure backend is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from rag.ingestion import ingest_document, get_collection_stats

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_docs")


def seed():
    files = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".txt")]

    if not files:
        print("No .txt files found in sample_docs/")
        return

    print(f"Found {len(files)} sample document(s). Ingesting...\n")

    total_chunks = 0
    for filename in files:
        path = os.path.join(SAMPLE_DIR, filename)
        try:
            n = ingest_document(path, filename)
            total_chunks += n
            print(f"  ✅  {filename} — {n} chunks stored")
        except Exception as e:
            print(f"  ❌  {filename} — Error: {e}")

    count, docs = get_collection_stats()
    print(f"\nDone. Total chunks in DB: {count}")
    print(f"Documents: {docs}")


if __name__ == "__main__":
    seed()
