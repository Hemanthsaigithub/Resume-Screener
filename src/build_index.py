"""
Builds a FAISS vector index over job description chunks so we can
retrieve the most relevant requirements text for a given query.

Input:
    data/postings.csv

Output:
    models/job_index.faiss
    models/job_chunks.pkl   (metadata: text + job_title for each chunk)
"""

import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

JOBS_PATH = "data/postings.csv"
INDEX_PATH = "models/job_index.faiss"
CHUNKS_PATH = "models/job_chunks.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
MAX_JOBS = 5000   # limit while developing; raise once the pipeline works
MIN_CHUNK_LEN = 20


def chunk_job_descriptions(jobs: pd.DataFrame) -> list:
    chunks = []
    for _, row in jobs.iterrows():
        desc = row.get("description")
        if not isinstance(desc, str):
            continue
        for line in desc.split("\n"):
            line = line.strip()
            if len(line) > MIN_CHUNK_LEN:
                chunks.append({
                    "text": line,
                    "job_title": row.get("title", "Unknown"),
                })
    return chunks


def main():
    print("Loading job postings...")
    jobs = pd.read_csv(JOBS_PATH, nrows=MAX_JOBS)
    jobs = jobs.dropna(subset=["description"])

    print("Chunking job descriptions...")
    chunks = chunk_job_descriptions(jobs)
    print(f"Built {len(chunks)} chunks")

    print(f"Embedding chunks with {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved chunk metadata to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()