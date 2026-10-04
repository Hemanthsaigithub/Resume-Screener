"""
Ties the ML classifier and the RAG retrieval layer together:

  1. Predict fit (shortlist/reject) for a resume using the trained classifier.
  2. Retrieve the most relevant job-requirement chunks from the FAISS index,
     filtered by seniority level detected from the query and job titles.
  3. Ask Gemini to explain the score, grounded in the retrieved requirements.

Run this after:
    src/feature_engineering.py
    src/train_classifier.py
    src/build_index.py
"""

import os
import re
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_PATH = "models/classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
INDEX_PATH = "models/job_index.faiss"
CHUNKS_PATH = "models/job_chunks.pkl"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"

# Ordered from most to least senior so we can match keyword -> level.
SENIORITY_PATTERNS = [
    ("senior", re.compile(r"(?i)\b(senior|sr\.?|lead|principal|staff)\b")),
    ("entry", re.compile(
        r"(?i)\b(junior|jr\.?|entry[\s\-]?level|intern|associate)\b")),
    ("mid", re.compile(r"(?i)\b(mid[\s\-]?level|ii|iii)\b")),
]


def detect_seniority(title: str) -> str:
    """
    Return 'senior', 'entry', 'mid', or 'unspecified' based on keywords
    found in a job title or query string. Checked in order so titles like
    'Senior Software Engineer II' still resolve to 'senior' first.
    """
    if not isinstance(title, str):
        return "unspecified"
    for level, pattern in SENIORITY_PATTERNS:
        if pattern.search(title):
            return level
    return "unspecified"


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return clf, vectorizer, index, chunks, embed_model


# Generic words to ignore when extracting the "domain" part of a query,
# so seniority terms and filler words don't count as domain keywords.
STOPWORDS = {
    "senior", "sr", "junior", "jr", "entry", "level", "mid", "lead",
    "principal", "staff", "the", "a", "an", "and", "or", "for", "of",
}


def extract_domain_keywords(text: str) -> set:
    """Pull out the meaningful (non-seniority, non-filler) words from a
    query or title, used to check that a retrieved chunk is actually
    about the right kind of role, not just the right seniority level."""
    if not isinstance(text, str):
        return set()
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def retrieve(query: str, index, chunks, embed_model, k: int = 5, min_after_filter: int = 3):
    """
    Retrieve the top-k chunks for a query, restricted to chunks whose job
    title matches the query's detected seniority level AND shares at least
    one domain keyword with the query (e.g. "software", "engineer"),
    so a query like "senior software" can't be satisfied by an unrelated
    "Senior Mechanical Engineer" posting just because the level matches.
    Falls back to unfiltered retrieval if too few matches survive, so the
    query never returns empty-handed.
    """
    query_level = detect_seniority(query)
    query_keywords = extract_domain_keywords(query)

    query_vec = embed_model.encode([query]).astype("float32")
    # Over-fetch so we have enough candidates left after filtering.
    distances, indices = index.search(np.array(query_vec), k * 10)
    candidates = [chunks[i] for i in indices[0] if i < len(chunks)]

    def domain_match(chunk_title: str) -> bool:
        if not query_keywords:
            return True  # nothing meaningful to match on, don't filter
        return bool(query_keywords & extract_domain_keywords(chunk_title))

    def level_match(chunk_title: str) -> bool:
        chunk_level = detect_seniority(chunk_title)
        if query_level == "unspecified":
            return chunk_level in ("entry", "unspecified")
        return chunk_level == query_level

    filtered = [
        c for c in candidates
        if domain_match(c["job_title"]) and level_match(c["job_title"])
    ]

    if len(filtered) >= min_after_filter:
        return filtered[:k]

    # Relax to domain-only match if seniority+domain was too strict.
    domain_only = [c for c in candidates if domain_match(c["job_title"])]
    if len(domain_only) >= min_after_filter:
        return domain_only[:k]

    # Last resort: unfiltered top-k rather than returning too little.
    return candidates[:k]


def fit_band(proba: float) -> str:
    """Map a fit probability to a human-friendly band instead of a stark binary label."""
    if proba >= 0.50:
        return "Strong Fit"
    elif proba >= 0.30:
        return "Possible Fit"
    else:
        return "Weak Fit"


def build_prompt(resume_text, job_title_query, band, proba, retrieved_chunks):
    context = "\n".join(f"- {c['text']}" for c in retrieved_chunks)
    return f"""A resume was scored {proba:.0%} fit for a role matching: "{job_title_query}"
Model verdict: {band}

Relevant job requirements retrieved for this role:
{context}

Resume excerpt (truncated):
{resume_text[:1000]}

Explain concisely (3-5 sentences) why this resume received this score,
referencing the specific requirements above where relevant. Then suggest
2-3 concrete, actionable improvements the candidate could make.
"""


def screen_resume(resume_text: str, job_title_query: str, clf, vectorizer, index, chunks, embed_model, client):
    # 1. Retrieve relevant job requirement chunks FIRST — these represent
    #    the actual job the resume is being screened against.
    retrieved_chunks = retrieve(
        job_title_query, index, chunks, embed_model, k=5)
    job_description_proxy = " ".join(c["text"] for c in retrieved_chunks)

    # 2. ML prediction — now conditioned on BOTH the resume and the job,
    #    not the resume alone. This is what makes the score role-specific.
    combined_text = f"RESUME: {resume_text} JOB: {job_description_proxy}"
    vec = vectorizer.transform([combined_text])
    proba = clf.predict_proba(vec)[0][1]
    band = fit_band(proba)

    # 3. Generation
    prompt = build_prompt(resume_text, job_title_query,
                          band, proba, retrieved_chunks)
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt)

    return {
        "prediction": band,
        "fit_probability": round(float(proba), 3),
        "retrieved_chunks": retrieved_chunks,
        "explanation": response.text,
    }


def main():
    clf, vectorizer, index, chunks, embed_model = load_artifacts()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Example
    sample_resume = (
        "Experienced data analyst skilled in Python, SQL, and Tableau. "
        "3 years building dashboards and running A/B tests."
    )
    sample_job_title = "Data Analyst"

    result = screen_resume(
        sample_resume, sample_job_title, clf, vectorizer, index, chunks, embed_model, client
    )

    print(
        f"Prediction: {result['prediction']} ({result['fit_probability']:.0%} fit)")
    print("\nRetrieved requirements:")
    for c in result["retrieved_chunks"]:
        print(f"  - {c['text']}")
    print("\nExplanation:\n", result["explanation"])


if __name__ == "__main__":
    main()
