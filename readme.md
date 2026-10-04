# Resume Screener + Advisor

A project combining a classic ML classifier with a RAG (Retrieval-Augmented
Generation) pipeline to screen resumes against specific job roles and explain
the score using real, retrieved job requirements.

- **ML component:** predicts a resume's fit score for a target role using a
  TF-IDF + Logistic Regression classifier trained on resume + job-description
  pairs.
- **RAG component:** retrieves the most relevant job-requirement snippets for
  the target role (filtered by seniority level and domain), and uses an LLM
  (Gemini) to generate a grounded explanation and improvement suggestions.

---
 
## Folder / File Structure

```
resume-screener/
├── data/
│   ├── Resume.csv          # Raw resume dataset (Kaggle: snehaanbhawal/resume-dataset)
│   ├── postings.csv        # Raw job postings dataset (Kaggle: arshkon/linkedin-job-postings)
│   └── pairs.csv           # Generated: resume-job pairs with heuristic fit labels
├── models/
│   ├── classifier.pkl      # Trained Logistic Regression model
│   ├── vectorizer.pkl      # Fitted TF-IDF vectorizer
│   ├── job_index.faiss     # FAISS vector index over job description chunks
│   └── job_chunks.pkl      # Metadata (text + job title) for each indexed chunk
├── src/
│   ├── feature_engineering.py  # Builds data/pairs.csv from the raw datasets
│   ├── train_classifier.py     # Trains the ML classifier on pairs.csv
│   ├── build_index.py          # Chunks job descriptions and builds the FAISS index
│   ├── rag_pipeline.py         # Ties ML + retrieval + Gemini generation together
│   └── app.py                  # Streamlit UI for the whole pipeline
├── venv/                   # Python virtual environment (not committed to git)
├── .env                     # Holds GEMINI_API_KEY (not committed to git)
├── .gitignore
├── requirements_updated.txt
└── README.md
```

### What each file actually does

**`src/feature_engineering.py`**
Loads `Resume.csv` and `postings.csv`, extracts a small skill vocabulary from
both resume text and job descriptions, pairs each resume with a matching job
posting (by category keyword), and computes a heuristic **fit score**
(skill-overlap ratio) used as a training label. Saves the result to
`data/pairs.csv`. This is a workaround for the fact that real hiring
shortlist/reject data isn't publicly available.

**`src/train_classifier.py`**
Trains a Logistic Regression model on TF-IDF features built from
`RESUME: {resume text} JOB: {job description}` (resume and job description
concatenated together, so the model learns resume-to-job fit, not just
generic resume quality). Saves `models/classifier.pkl` and
`models/vectorizer.pkl`.

**`src/build_index.py`**
Chunks every job description into individual lines, filters out short/
header-style lines (e.g. `"TITLE: Senior Software Engineer"`) that add no
real requirement content, deduplicates repeated boilerplate lines, embeds
the remaining chunks with `sentence-transformers` (`all-MiniLM-L6-v2`), and
builds a FAISS index for similarity search. Saves `models/job_index.faiss`
and `models/job_chunks.pkl`.

**`src/rag_pipeline.py`**
The core pipeline, run for a given `(resume_text, job_title_query)` pair:

1. Detects the seniority level and domain keywords implied by the query
   (e.g. "senior" vs "junior"; "software" vs "mechanical").
2. Retrieves the most relevant job-requirement chunks from the FAISS index,
   filtered to match both seniority and domain — falling back to looser
   filtering if too few chunks survive.
3. Feeds the resume + retrieved job content into the trained classifier to
   get a fit probability, mapped to a **Strong Fit / Possible Fit / Weak
   Fit** band (rather than a stark binary label).
4. Sends a prompt with the score, band, and retrieved chunks to Gemini to
   generate a grounded explanation and concrete improvement suggestions.

**`src/app.py`**
A Streamlit UI wrapping `rag_pipeline.py` — paste a resume, type a target
job title, click "Screen Resume," and see the fit band, score, retrieved
requirements, and AI-generated explanation.

---

## Setup — from a clean machine

**1. Get the code and data folder ready**

```
mkdir resume-screener
cd resume-screener
```

Create the folder structure above (`data/`, `models/`, `src/`) and place the
provided `.py` files into `src/`.

**2. Create and activate a virtual environment**

```
python -m venv venv
venv\Scripts\activate        # Windows
```

**3. Install dependencies**

```
pip install -r requirements.txt
```

**4. Download the datasets**

- Resume dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
  → extract `Resume.csv` into `data/`
- Job postings dataset: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
  → extract `postings.csv` into `data/`

Sanity-check both files load correctly before proceeding:

```
python -c "import pandas as pd; print(pd.read_csv('data/Resume.csv', nrows=1).columns.tolist())"
python -c "import pandas as pd; print(pd.read_csv('data/postings.csv', nrows=1).columns.tolist())"
```

**5. Get a free Gemini API key**

- Go to https://aistudio.google.com/app/apikey and create a key.
- Create a `.env` file in the project root:
  ```
  GEMINI_API_KEY=your_key_here
  ```

---

## Running the pipeline (in order)

Each step depends on the output of the previous one — run them in this order
the first time:

```
# 1. Build labeled resume-job pairs
python src/feature_engineering.py

# 2. Train the ML classifier
python src/train_classifier.py

# 3. Build the RAG retrieval index (takes several minutes — embeds ~50k chunks)
python src/build_index.py

# 4. Sanity-check the full pipeline in the terminal (no UI)
python src/rag_pipeline.py

# 5. Launch the actual app
streamlit run src/app.py
```

Steps 1–3 only need to be re-run if you change the underlying data or the
feature engineering / chunking logic. Once `models/` is populated, you can
go straight to step 5 for everyday use.

---

## Limitations (and why they exist)

**1. Training labels are heuristic, not real hiring decisions.**
Real shortlist/reject data from actual hiring pipelines isn't publicly
available, so `feature_engineering.py` approximates a fit label using
skill-keyword overlap between a resume and a matched job posting. This
means the classifier is learning a _proxy_ for fit, not ground truth — a
resume can be genuinely strong yet score low if it doesn't share many
literal keywords with the paired job posting.

**2. TF-IDF only understands literal word overlap, not meaning.**
The classifier has no concept that "Python" relates to "programming," or
that "led a team of 5" implies seniority. It purely counts which exact
words appear in both the resume and the job text. This makes the fit score
sensitive to phrasing rather than actual qualification — a well-qualified
candidate using different terminology than the job posting can score
lower than expected. A transformer-based embedding classifier would
capture semantic similarity instead of literal overlap, at the cost of
more complexity and compute.

**3. Vector retrieval (FAISS + sentence-transformers) ranks by semantic
closeness, not hard constraints.**
Embedding similarity is a _soft_ measure — "Senior Software Engineer" and
"Senior Mechanical Engineer" can embed close together because both are
about seniority and engineering, even though the domains are unrelated.
Pure vector search has no built-in way to enforce constraints like "must
match this exact seniority level" or "must be this domain." This project
works around it with manual post-retrieval filtering (`detect_seniority()`
and `extract_domain_keywords()` in `rag_pipeline.py`), with a fallback to
looser filtering when too few chunks survive. This is a reasonable patch,
but a production system would more likely use hybrid search (keyword +
vector) with metadata filters applied _before_ the vector search, not
after.

**4. Seniority and domain detection are keyword-based, not learned.**
`detect_seniority()` and `extract_domain_keywords()` rely on regex/keyword
matching against job titles. This misses roles that don't explicitly state
their level or domain in the title (e.g. a senior-level role without the
word "senior" anywhere), and can misfire on unusual title phrasing.

**5. Job description chunking is line-based, which is noisy.**
Chunks are built by splitting on newlines and filtering short/header-like
lines. Some postings format their text as a single unbroken paragraph or
with inconsistent line breaks, so chunk quality (and therefore retrieval
quality) varies across postings depending on how the original posting was
formatted.

**6. Small training set.**
Only ~2,484 resumes are available, each paired with a small number of jobs.
This limits how well the classifier can generalize across the full variety
of resume phrasing and job requirement language.

**Taken together:** every stage of this pipeline (labels → classifier →
retrieval → filtering) is a reasonable simplification for a learning
project, but each has known blind spots that compound.
