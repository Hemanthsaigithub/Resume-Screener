"""
Streamlit UI for the Resume Screener + Advisor.

Run with:
    streamlit run src/app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

from rag_pipeline import load_artifacts, screen_resume

load_dotenv()

st.set_page_config(page_title="Resume Screener + Advisor", page_icon="📄")
st.title("📄 Resume Screener + Advisor")
st.write(
    "Paste a resume and a target job title. The ML model predicts a fit "
    "score, and the RAG layer explains the result using real job "
    "requirements retrieved from a job postings dataset."
)


@st.cache_resource
def get_artifacts():
    clf, vectorizer, index, chunks, embed_model = load_artifacts()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return clf, vectorizer, index, chunks, embed_model, client


clf, vectorizer, index, chunks, embed_model, client = get_artifacts()

resume_text = st.text_area("Resume text", height=250, placeholder="Paste resume text here...")
job_title_query = st.text_input("Target job title", placeholder="e.g. Data Analyst")

if st.button("Screen Resume", type="primary"):
    if not resume_text.strip() or not job_title_query.strip():
        st.warning("Please provide both a resume and a job title.")
    else:
        with st.spinner("Scoring resume and retrieving relevant requirements..."):
            result = screen_resume(
                resume_text, job_title_query, clf, vectorizer, index, chunks, embed_model, client
            )

        st.subheader("Result")
        col1, col2 = st.columns(2)
        col1.metric("Prediction", result["prediction"])
        col2.metric("Fit score", f"{result['fit_probability']:.0%}")

        st.subheader("Retrieved job requirements")
        for c in result["retrieved_chunks"]:
            st.markdown(f"- {c['text']}")

        st.subheader("Explanation & suggestions")
        st.write(result["explanation"])