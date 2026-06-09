"""
pipeline.py - The Cinematic Pipeline (replaces agent.py).

WHAT THIS DOES (plain English):
================================
You give it a movie plot idea. It does 4 things:

    1. Checks if your idea is too similar to an existing movie  (local math, no AI)
    2. Finds which famous director made the closest matching film (database lookup)
    3. Rewrites your plot in that director's filmmaking style     (Ollama/Gemma3)
    4. Packages everything into validated, structured JSON        (Pydantic)


PROCESS FLOW:
=============

    [User types a plot idea]
            |
            v
    +------------------+
    |  STEP 1:         |   TF-IDF turns plots into word-importance scores.
    |  Plagiarism      |   Cosine similarity measures overlap.
    |  Detection       |   Score >= 0.30 --> "too similar" flag.
    |                  |
    |  (scikit-learn)  |   NO API calls. NO neural networks.
    |  (runs locally)  |   Fully deterministic - same input = same output.
    +--------+---------+
             |  closest movie + director + similarity score
             v
    +------------------+
    |  STEP 2:         |   The matched movie tells us the director.
    |  Director        |   e.g. matched "Inception" --> Christopher Nolan.
    |  Routing         |   This is a simple column lookup, not AI.
    +--------+---------+
             |  director name
             v
    +------------------+
    |  STEP 3:         |   Gemma3 via Ollama rewrites the user's plot
    |  LLM Style       |   in the matched director's filmmaking style.
    |  Rewrite         |
    |  (Ollama (local))    |   THIS IS THE ONLY STEP USING AI. Runs locally.
    +--------+---------+
             |  raw JSON text from LLM
             v
    +------------------+
    |  STEP 4:         |   Pydantic checks every field:
    |  Validation      |   correct types? all present? valid values?
    |  (Pydantic)      |   If anything is wrong --> hard error, no silent failures.
    +------------------+
             |
             v
    [Validated MovieAnalysis JSON output]


BUSINESS USE CASE:
==================
A screenwriter or development executive pastes a plot idea. The system:
    1. Flags legal risk    - "Your plot is 72% similar to The Martian"
    2. Sparks creativity   - "Here's your plot as Wes Anderson would direct it"
    3. Structures output   - clean JSON that downstream tools can consume

"""

import os
import json
import logging
from datetime import datetime

import pandas as pd
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from schema import MovieAnalysis, PLAGIARISM_THRESHOLD

# Which Ollama model to use for the style rewrite.
# Change this if you pull a different model (e.g. "llama3", "mistral").
OLLAMA_MODEL = "gemma3"


# -----------------------------------------------------------------------
# Logging - every run writes a human-readable log
# -----------------------------------------------------------------------

def setup_logger(name: str = "pipeline") -> logging.Logger:
    """
    Create a logger that writes to both the console AND a timestamped file.

    Log files go to logs/<name>_<timestamp>.log so you can always go back
    and see exactly what happened on any given run.
    """
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/{name}_{timestamp}.log"

    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.log_path = log_path  # stash path for later reference
    return logger


# -----------------------------------------------------------------------
# STEP 1: Plagiarism Detection (deterministic, local, no API)
# -----------------------------------------------------------------------

def detect_plagiarism(user_plot: str, df: pd.DataFrame) -> dict:
    """
    Compare the user's plot against every movie in the database using TF-IDF.

    HOW TF-IDF WORKS (plain English):
        - Each plot becomes a list of word-importance scores.
        - Common filler words ("the", "a", "is") get near-zero scores.
        - Distinctive words ("cyborg", "heist", "wormhole") get high scores.
        - Cosine similarity then measures how much two plots share
          those distinctive words.
        - 1.0 = identical wording.  0.0 = nothing in common.

    Args:
        user_plot: The user's free-text plot description.
        df:        DataFrame with at least columns: title, director, plot.

    Returns:
        dict with:
            matched_movie      - title of the closest film
            assigned_director   - that film's director
            similarity_score    - cosine similarity (0.0 - 1.0)
            detected_plagiarism - True if score >= PLAGIARISM_THRESHOLD
            top_matches         - list of top 5 matches for transparency
    """
    db_plots = df["plot"].tolist()

    # Build TF-IDF matrix: all database plots + user's plot at the end
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(db_plots + [user_plot])

    # Compare user (last row) against all database entries
    user_vector = tfidf_matrix[-1]
    db_vectors = tfidf_matrix[:-1]
    scores = sklearn_cosine(user_vector, db_vectors).flatten()

    # Best match
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    # Top 5 for transparency logging
    top5_idx = scores.argsort()[-5:][::-1]
    top_matches = [
        {
            "rank": rank,
            "title": df.iloc[idx]["title"],
            "director": df.iloc[idx]["director"],
            "score": round(float(scores[idx]), 4),
        }
        for rank, idx in enumerate(top5_idx, 1)
    ]

    return {
        "matched_movie": df.iloc[best_idx]["title"],
        "assigned_director": df.iloc[best_idx]["director"],
        "similarity_score": round(best_score, 4),
        "detected_plagiarism": best_score >= PLAGIARISM_THRESHOLD,
        "top_matches": top_matches,
    }


# -----------------------------------------------------------------------
# STEP 2 + 3: Director Routing --> LLM Style Rewrite
# -----------------------------------------------------------------------

def rewrite_in_director_style(
    user_plot: str,
    matched_movie: str,
    assigned_director: str,
    similarity_score: float,
    detected_plagiarism: bool,
) -> str:
    """
    Send the user's plot + detection results to a local LLM for style rewriting.

    The LLM receives hard facts from Step 1 (score, match, plagiarism flag)
    with strict instructions NOT to override them. Its only creative job is
    the rewrite and the stylistic notes.

    Runs locally via Ollama — no API key, no network calls, no rate limits.

    Returns:
        Raw JSON string from the LLM (to be validated in Step 4).
    """
    prompt = f"""You are an expert film critic and professor of cinematography.

USER'S ORIGINAL PLOT:
"{user_plot}"

PLAGIARISM DETECTION RESULTS (computed deterministically by TF-IDF — do NOT change these):
- Closest match: "{matched_movie}"
- Director: {assigned_director}
- Similarity score: {similarity_score}
- Plagiarism detected: {detected_plagiarism}

YOUR TASK — respond with ONLY a JSON object, no other text:
1. Set "detected_plagiarism" to exactly {str(detected_plagiarism).lower()}.
2. Set "matched_movie" to exactly "{matched_movie}".
3. Set "similarity_score" to exactly {similarity_score}.
4. Set "assigned_director" to exactly "{assigned_director}".
5. In "rewritten_plot", rewrite the user's plot in {assigned_director}'s filmmaking style.
   Adopt their narrative structure, pacing, tone, and thematic obsessions.
6. In "stylistic_notes", explain in plain English what you changed and why it
   reflects {assigned_director}'s style. A non-film-expert should understand this.

Required JSON schema:
{{
  "detected_plagiarism": bool,
  "matched_movie": string,
  "similarity_score": float,
  "assigned_director": string,
  "rewritten_plot": string,
  "stylistic_notes": string
}}"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.1},
    )
    return response.message.content


# -----------------------------------------------------------------------
# STEP 4: Pydantic Validation
# -----------------------------------------------------------------------

def validate_output(raw_json: str) -> MovieAnalysis:
    """
    Parse the LLM's JSON and validate every field against the schema.

    This is the quality gate. If the LLM returned malformed JSON, missing
    fields, or wrong types, this raises a clear error rather than letting
    garbage through silently.

    Returns:
        Validated MovieAnalysis instance.

    Raises:
        json.JSONDecodeError if the string isn't valid JSON.
        pydantic.ValidationError if fields are missing/wrong type.
    """
    data = json.loads(raw_json)
    return MovieAnalysis.model_validate(data)


# -----------------------------------------------------------------------
# FULL PIPELINE: orchestrates all 4 steps with logging
# -----------------------------------------------------------------------

def run_pipeline(
    user_plot: str,
    csv_path: str = "movies_dataset.csv",
    log: logging.Logger | None = None,
    df: pd.DataFrame | None = None,
    detection: dict | None = None,
) -> MovieAnalysis:
    """
    Run the full cinematic pipeline end-to-end.

    Args:
        user_plot:  Free-text plot description from the user.
        csv_path:   Path to the movie database CSV (used only if df is None).
        log:        Optional logger. If None, a new one is created.
        df:         Optional pre-loaded DataFrame. Avoids re-reading CSV.
        detection:  Optional pre-computed detection result from detect_plagiarism().
                    Avoids running TF-IDF twice when the caller already has it.

    Returns:
        Validated MovieAnalysis object.

    Every step is logged to both console and a file in logs/.
    """
    if log is None:
        log = setup_logger("pipeline")

    log.info("=" * 60)
    log.info("CINEMATIC PIPELINE - NEW RUN")
    log.info("=" * 60)
    log.info(f"Input: \"{user_plot[:120]}{'...' if len(user_plot) > 120 else ''}\"")

    # Load database (skip if caller passed a DataFrame)
    if df is None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Movie database not found at: {csv_path}")
        df = pd.read_csv(csv_path)
    log.info(f"Database: {len(df)} movies")

    # -- STEP 1: Plagiarism detection (local) --
    log.info("-" * 50)
    log.info("STEP 1: TF-IDF Plagiarism Detection (local)")
    if detection is None:
        detection = detect_plagiarism(user_plot, df)

    is_plag = detection["detected_plagiarism"]
    log.info(f"  Best match:  \"{detection['matched_movie']}\"")
    log.info(f"  Director:    {detection['assigned_director']}")
    log.info(f"  Score:       {detection['similarity_score']}")
    log.info(f"  Plagiarism:  {'YES' if is_plag else 'NO'}  (threshold: {PLAGIARISM_THRESHOLD})")
    log.info("  Top 5 matches:")
    for m in detection["top_matches"]:
        log.info(f"    {m['rank']}. \"{m['title']}\" ({m['director']}) - {m['score']}")

    # -- STEP 2: Director routing --
    log.info("-" * 50)
    log.info(f"STEP 2: Routed to director -> {detection['assigned_director']}")

    # -- STEP 3: LLM rewrite --
    log.info("STEP 3: Calling Gemma3 via Ollama for style rewrite...")
    raw_json = rewrite_in_director_style(
        user_plot=user_plot,
        matched_movie=detection["matched_movie"],
        assigned_director=detection["assigned_director"],
        similarity_score=detection["similarity_score"],
        detected_plagiarism=detection["detected_plagiarism"],
    )
    log.info("  LLM response received.")

    # -- STEP 4: Pydantic validation --
    log.info("-" * 50)
    log.info("STEP 4: Pydantic schema validation")
    result = validate_output(raw_json)
    log.info("  Validation: PASSED")
    log.info(f"  Fields: {list(result.model_dump().keys())}")

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)

    return result


# -----------------------------------------------------------------------
# BONUS: Differentiation Strategies
# -----------------------------------------------------------------------

def suggest_differentiation(
    user_plot: str,
    matched_movie: str,
    assigned_director: str,
    similarity_score: float,
) -> list[dict]:
    """
    When a plot is flagged (or borderline), suggest specific changes
    to make it more original.

    This is the creative consulting step: instead of just saying
    "you copied The Martian," it says "here's how to make it yours."

    Returns:
        List of dicts, each with:
            strategy:    Short name (e.g. "Shift the setting")
            description: What to change and why it creates distance
    """
    prompt = f"""You are a creative writing consultant helping a screenwriter
make their plot idea more original.

THEIR PLOT:
"{user_plot}"

PROBLEM:
This is {similarity_score:.0%} similar to "{matched_movie}" (directed by {assigned_director}).

YOUR TASK:
Suggest 3 to 5 specific, actionable changes that would make this plot
meaningfully different from "{matched_movie}" while keeping the core
idea intact. Each suggestion should create real distance from the
existing film, not just surface-level changes.

Respond with ONLY a JSON object:
{{
  "strategies": [
    {{"strategy": "short name", "description": "1-2 sentences explaining the change and why it helps"}},
    ...
  ]
}}"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.7},
    )

    data = json.loads(response.message.content)
    return data.get("strategies", [])


# -----------------------------------------------------------------------
# Standalone execution
# -----------------------------------------------------------------------

if __name__ == "__main__":
    sample = "An astronaut gets isolated on a desert planet and has to grow food to survive."
    print("\nRunning Cinematic Pipeline...\n")
    result = run_pipeline(sample)
    print("\nFinal Output:")
    print(json.dumps(result.model_dump(), indent=2))
