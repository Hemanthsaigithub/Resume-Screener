"""
Trains a Logistic Regression classifier to predict resume-job fit
(shortlist vs reject) for a SPECIFIC role — not a generic resume score.

The model input is the resume text CONCATENATED with the job description,
so the same resume can score differently against different roles.

Input:
    data/pairs.csv (produced by feature_engineering.py)

Output:
    models/classifier.pkl
    models/vectorizer.pkl
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

PAIRS_PATH = "data/pairs.csv"
MODEL_PATH = "models/classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def main():
    pairs = pd.read_csv(PAIRS_PATH)
    pairs = pairs.dropna(subset=["resume_text", "job_description"])
    print(f"Loaded {len(pairs)} pairs")
    print(pairs["label"].value_counts())

    # Combine resume + job description so the model learns resume-to-JOB fit, not just generic resume quality.
    pairs["combined_text"] = (
        "RESUME: " + pairs["resume_text"] + " JOB: " + pairs["job_description"]
    )

    X = pairs["combined_text"]
    y = pairs["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print("\nClassification report:")
    print(classification_report(y_test, preds))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
