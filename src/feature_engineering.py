"""
Feature engineering for the Resume Screener + Advisor project.

Loads the resume dataset and job postings dataset, extracts a simple
skill vocabulary from both, pairs resumes with jobs (matched by category
keyword), and computes a heuristic fit score used as the training label
for the classifier.

Input files expected:
    data/Resume.csv    -> columns: id, resume_str, resume_html, category
    data/postings.csv  -> columns include: job_title, company_name, job_location, job_description, salary, post_date

Output:
    data/pairs.csv -> resume_text, job_title, job_description, fit_score, label
"""

import re
import pandas as pd

RESUME_PATH = "data/Resume.csv"
JOBS_PATH = "data/postings.csv"
OUTPUT_PATH = "data/pairs.csv"

SKILL_VOCAB = [
    "python", "sql", "excel", "java", "javascript", "react", "aws",
    "machine learning", "deep learning", "communication",
    "project management", "leadership", "data analysis", "tableau",
    "power bi", "git", "docker", "kubernetes", "c++", "html", "css",
    "marketing", "sales", "customer service", "accounting", "nursing",
]

FIT_THRESHOLD = 0.4  # overlap ratio >= this -> label as "shortlist"


def extract_skills(text: str, vocab=SKILL_VOCAB) -> set:
    """Lowercase the text and return the set of vocab skills found in it."""
    if not isinstance(text, str):
        return set()
    text = text.lower()
    return {skill for skill in vocab if skill in text}


def fit_score(resume_skills: set, job_skills: set) -> float:
    """Fraction of the job's required skills that appear in the resume."""
    if not job_skills:
        return 0.0
    overlap = len(resume_skills & job_skills)
    return overlap / len(job_skills)


def build_pairs(resumes: pd.DataFrame, jobs: pd.DataFrame, jobs_per_resume: int = 1) -> pd.DataFrame:
    """
    Pair each resume with one or more job postings whose title contains a keyword from the resume's category, then compute a fit score.
    """
    jobs = jobs.dropna(subset=["description", "title"]).reset_index(drop=True)
    jobs["job_skills"] = jobs["description"].apply(extract_skills)

    rows = []
    for _, r in resumes.iterrows():
        resume_skills = extract_skills(r["Resume_str"])
        category_keyword = str(r["Category"]).split()[0].lower()

        # Find jobs whose title loosely matches the resume's category.
        matches = jobs[jobs["title"].str.lower().str.contains(category_keyword, na=False)]
        if matches.empty:
            matches = jobs.sample(min(jobs_per_resume, len(jobs)), random_state=42)
        else:
            matches = matches.sample(min(jobs_per_resume, len(matches)), random_state=42)

        for _, j in matches.iterrows():
            score = fit_score(resume_skills, j["job_skills"])
            rows.append({
                "resume_text": r["Resume_str"],
                "category": r["Category"],
                "job_title": j["title"],
                "job_description": j["description"],
                "fit_score": score,
                "label": int(score >= FIT_THRESHOLD),
            })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    resumes = pd.read_csv(RESUME_PATH)
    jobs = pd.read_csv(JOBS_PATH)

    print(f"Resumes: {resumes.shape}, Jobs: {jobs.shape}")

    pairs = build_pairs(resumes, jobs, jobs_per_resume=1)
    pairs.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(pairs)} resume-job pairs to {OUTPUT_PATH}")
    print(pairs["label"].value_counts())


if __name__ == "__main__":
    main()