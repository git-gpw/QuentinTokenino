"""
pipeline.py - The Cinematic Pipeline.

WHAT THIS DOES (plain English):
================================
You give it a movie plot idea. It does 4 things:

    1. Checks if your idea is too similar to an existing movie  (SBERT embeddings, local)
    2. Finds which famous director made the closest matching film (database lookup)
    3. Rewrites your plot in that director's filmmaking style     (Ollama/Gemma3)
    4. Packages everything into validated, structured JSON        (Pydantic)


PROCESS FLOW:
=============

    [User types a plot idea]
            |
            v
    +------------------+
    |  STEP 1:         |   Dual-signal: SBERT (semantic meaning) + TF-IDF
    |  Plagiarism      |   (vocabulary overlap), combined via geometric mean.
    |  Detection       |   Score >= threshold --> "too similar" flag.
    |                  |   Popularity-weighted ranking surfaces famous films.
    |  (SBERT + TF-IDF)|   NO API calls. Runs locally on CPU.
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
import hashlib
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import ollama
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from schema import MovieAnalysis, PLAGIARISM_THRESHOLD

# Load models once at import time
_nlp = spacy.load("en_core_web_sm")         # NER + noun chunks (~12MB)
_SBERT = SentenceTransformer("all-MiniLM-L6-v2")  # Semantic embeddings (~80MB)

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

    Cleanup: call cleanup_logger(logger) when done to close file handlers
    and prevent file descriptor leaks in long-running processes.
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


def cleanup_logger(logger: logging.Logger):
    """Close and remove all handlers to prevent file descriptor leaks."""
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


# -----------------------------------------------------------------------
# STEP 1: Plagiarism Detection (deterministic, local, no API)
# -----------------------------------------------------------------------

# ---- Scoring config ----
# Dual-signal approach: SBERT captures semantic meaning (paraphrases),
# TF-IDF captures vocabulary overlap (distinguishes genre similarity from
# actual plagiarism). Combined via geometric mean: sqrt(sbert * tfidf).
# This requires BOTH signals to be high for a high combined score.
POPULARITY_BOOST = 0.15  # max boost for ranking (not for threshold decision)


def _precompute_embeddings(df: pd.DataFrame) -> np.ndarray:
    """
    Encode all movie plots into 384-dim sentence embeddings using SBERT.

    Called once at startup. Returns an (N, 384) float32 matrix.
    ~2-3 minutes for 16k plots on CPU, cached to disk afterward.
    """
    plots = df["plot"].tolist()
    embeddings = _SBERT.encode(plots, batch_size=256, show_progress_bar=True,
                               normalize_embeddings=True)
    return embeddings


def _prefit_tfidf(df: pd.DataFrame):
    """Fit TF-IDF vectorizer on movie plots and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(stop_words="english")
    db_matrix = vectorizer.fit_transform(df["plot"].tolist())
    return vectorizer, db_matrix


# Module-level cache — populated by app.py at startup via init_nlp()
_DB_EMBEDDINGS = None
_TFIDF_VECTORIZER = None
_TFIDF_MATRIX = None
_DB_POPULARITY = None

# Disk cache location
_NLP_CACHE_PATH = "nlp_cache.pkl"


def _csv_hash(df: pd.DataFrame) -> str:
    """Fast hash of DataFrame content to detect dataset changes."""
    content = "".join(df["plot"].tolist()).encode("utf-8")
    return hashlib.sha256(content).hexdigest()[:16]


def _load_cache(expected_hash: str):
    """Load cached NLP features from disk if hash matches."""
    if not os.path.exists(_NLP_CACHE_PATH):
        return None
    try:
        with open(_NLP_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("hash") == expected_hash:
            return cache
        else:
            print(f"  NLP cache hash mismatch — dataset changed, will recompute")
    except (pickle.UnpicklingError, EOFError, KeyError) as e:
        print(f"  WARNING: NLP cache file is corrupted ({type(e).__name__}), will recompute")
    return None


def _save_cache(data_hash, embeddings, tfidf_vectorizer, tfidf_matrix):
    """Save NLP features to disk for fast subsequent startups.

    Writes to a temp file first, then renames atomically to avoid
    leaving a corrupted cache if the process is interrupted mid-write.
    """
    cache = {
        "hash": data_hash,
        "embeddings": embeddings,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
    }
    tmp_path = _NLP_CACHE_PATH + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, _NLP_CACHE_PATH)


def init_nlp(df: pd.DataFrame):
    """
    Compute and cache all NLP features for the movie database. Call once at startup.

    On first run, encodes all plots with SBERT + TF-IDF and saves to
    nlp_cache.pkl (~3 min). On subsequent runs, loads from cache in
    ~2 seconds if the dataset hasn't changed (SHA-256 hash check).

    Caches:
        - SBERT embeddings (384-dim per movie, semantic similarity)
        - TF-IDF vectorizer + matrix (vocabulary overlap)
        - Popularity scores (for ranking boost)
    """
    global _DB_EMBEDDINGS, _TFIDF_VECTORIZER, _TFIDF_MATRIX, _DB_POPULARITY

    data_hash = _csv_hash(df)
    cache = _load_cache(data_hash)

    if cache is not None:
        print(f"  NLP cache hit ({_NLP_CACHE_PATH}) — loading pre-computed features")
        _DB_EMBEDDINGS = cache["embeddings"]
        _TFIDF_VECTORIZER = cache["tfidf_vectorizer"]
        _TFIDF_MATRIX = cache["tfidf_matrix"]
    else:
        print(f"  NLP cache miss — encoding {len(df)} movies with SBERT + TF-IDF...")
        _DB_EMBEDDINGS = _precompute_embeddings(df)
        _TFIDF_VECTORIZER, _TFIDF_MATRIX = _prefit_tfidf(df)
        _save_cache(data_hash, _DB_EMBEDDINGS, _TFIDF_VECTORIZER, _TFIDF_MATRIX)
        print(f"  NLP features cached to {_NLP_CACHE_PATH}")

    if "popularity" in df.columns:
        _DB_POPULARITY = df["popularity"].values
    else:
        _DB_POPULARITY = None


# Keep backward compat alias
init_tfidf = init_nlp


def detect_plagiarism(user_plot: str, df: pd.DataFrame) -> dict:
    """
    Compare the user's plot against every movie in the database using
    dual-signal similarity: SBERT (semantic) + TF-IDF (vocabulary).

    WHY TWO SIGNALS:
        - SBERT alone catches paraphrases but can't distinguish "same story"
          from "same genre" — both score high semantically.
        - TF-IDF alone misses paraphrases that use different words.
        - Combined via geometric mean: sqrt(sbert * tfidf). This requires
          BOTH signals to be high, naturally filtering genre overlap while
          catching true paraphrases.

    POPULARITY RANKING:
        When two movies score similarly, the more famous one ranks higher.
        Popularity affects *which* movie is shown, NOT the plagiarism decision.

    Args:
        user_plot: The user's free-text plot description.
        df:        DataFrame with columns: title, director, plot, [popularity].

    Returns:
        dict with:
            matched_movie       - title of the closest film
            assigned_director   - that film's director
            similarity_score    - geometric mean of SBERT + TF-IDF (0.0 - 1.0)
            detected_plagiarism - True if score >= PLAGIARISM_THRESHOLD
            top_matches         - list of top 5 matches for transparency
    """
    # ---- Signal 1: SBERT semantic similarity ----
    if _DB_EMBEDDINGS is not None:
        user_emb = _SBERT.encode([user_plot], normalize_embeddings=True)
        sbert_scores = sklearn_cosine(user_emb, _DB_EMBEDDINGS).flatten()
    else:
        all_plots = df["plot"].tolist() + [user_plot]
        all_emb = _SBERT.encode(all_plots, normalize_embeddings=True)
        sbert_scores = sklearn_cosine(all_emb[-1:], all_emb[:-1]).flatten()

    # ---- Signal 2: TF-IDF vocabulary overlap ----
    if _TFIDF_VECTORIZER is not None and _TFIDF_MATRIX is not None:
        user_vec = _TFIDF_VECTORIZER.transform([user_plot])
        tfidf_scores = sklearn_cosine(user_vec, _TFIDF_MATRIX).flatten()
    else:
        plots = df["plot"].tolist()
        vec = TfidfVectorizer(stop_words="english")
        db_matrix = vec.fit_transform(plots)
        user_vec = vec.transform([user_plot])
        tfidf_scores = sklearn_cosine(user_vec, db_matrix).flatten()

    # ---- Geometric mean: requires BOTH signals to be high ----
    # Clamp negatives to 0 before sqrt (rare but possible with SBERT)
    sbert_clamped = np.maximum(sbert_scores, 0)
    tfidf_clamped = np.maximum(tfidf_scores, 0)
    combined_scores = np.sqrt(sbert_clamped * tfidf_clamped)

    # ---- Popularity-boosted ranking score ----
    if _DB_POPULARITY is not None:
        ranking_scores = combined_scores * (1 + POPULARITY_BOOST * _DB_POPULARITY)
    elif "popularity" in df.columns:
        ranking_scores = combined_scores * (1 + POPULARITY_BOOST * df["popularity"].values)
    else:
        ranking_scores = combined_scores

    # Best match by ranking score (popularity-aware)
    best_idx = int(ranking_scores.argmax())
    # Use combined score (without popularity) for plagiarism decision
    best_score = float(combined_scores[best_idx])

    # Top 5 by ranking score
    top5_idx = ranking_scores.argsort()[-5:][::-1]
    top_matches = [
        {
            "rank": rank,
            "title": df.iloc[idx]["title"],
            "director": df.iloc[idx]["director"],
            "score": round(float(combined_scores[idx]), 4),
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

PLAGIARISM DETECTION RESULTS (computed deterministically — do NOT change these):
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
        options={"temperature": 0.1, "num_predict": 2048},
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
                    Avoids running detection twice when the caller already has it.

    Returns:
        Validated MovieAnalysis object.

    Every step is logged to both console and a file in logs/.
    """
    owns_logger = log is None
    if owns_logger:
        log = setup_logger("pipeline")

    try:
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
        log.info("STEP 1: SBERT Plagiarism Detection (local)")
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

    finally:
        if owns_logger:
            cleanup_logger(log)


# -----------------------------------------------------------------------
# BONUS: Similarity Explanation
# -----------------------------------------------------------------------

def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Use spaCy NER to extract named entities from a plot, grouped by type.

    This gives a deterministic, explainable foundation for similarity
    analysis — we can say "both plots mention NASA and Mars" rather than
    relying entirely on the LLM's interpretation.

    Returns:
        dict mapping entity type labels to lists of unique entity texts.
        e.g. {"PERSON": ["Mark Watney"], "ORG": ["NASA"], "GPE": ["Mars"]}
    """
    doc = _nlp(text)
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    return entities


def find_shared_entities(user_plot: str, matched_plot: str) -> list[dict]:
    """
    Compare spaCy NER output from both plots to find shared entities.

    Uses case-insensitive matching and substring containment to catch
    partial overlaps (e.g. "NASA" matches "NASA mission").

    Returns:
        List of dicts with:
            entity:   The shared entity text
            type:     NER label (PERSON, ORG, GPE, etc.)
            in_user:  How it appears in the user's plot
            in_match: How it appears in the matched plot
    """
    user_ents = extract_entities(user_plot)
    match_ents = extract_entities(matched_plot)

    shared = []
    seen = set()

    for ent_type, user_vals in user_ents.items():
        match_vals = match_ents.get(ent_type, [])
        for u in user_vals:
            for m in match_vals:
                u_low, m_low = u.lower(), m.lower()
                # Exact or substring match
                if u_low == m_low or u_low in m_low or m_low in u_low:
                    key = (ent_type, min(u_low, m_low))
                    if key not in seen:
                        seen.add(key)
                        shared.append({
                            "entity": u if len(u) >= len(m) else m,
                            "type": ent_type,
                            "in_user": u,
                            "in_match": m,
                        })
    return shared


def explain_similarity(
    user_plot: str,
    matched_movie: str,
    matched_plot: str,
    similarity_score: float,
) -> dict:
    """
    Combine spaCy NER (deterministic) + Ollama (creative) to explain
    what's most similar between the user's plot and the closest match.

    Two layers of analysis:
        1. NER overlap:  Extracted by spaCy — shared names, places, orgs.
                         Deterministic, reproducible, explainable.
        2. LLM analysis: Ollama identifies deeper thematic/structural
                         parallels, grounded by the NER findings.

    Args:
        user_plot:        The user's submitted plot.
        matched_movie:    Title of the closest matching movie.
        matched_plot:     The actual plot text of that movie from the database.
        similarity_score: Cosine similarity from TF-IDF.

    Returns:
        dict with:
            shared_entities: list of shared NER entities (deterministic)
            aspects:         list of thematic aspects from LLM
    """
    # ── Layer 1: Deterministic NER overlap ──
    shared = find_shared_entities(user_plot, matched_plot)

    # Format NER findings for the LLM prompt
    if shared:
        ner_context = "Named Entity Recognition found these shared elements:\n"
        for s in shared:
            ner_context += f"  - {s['entity']} ({s['type']})\n"
    else:
        ner_context = "Named Entity Recognition found no directly shared names/places/organizations.\n"

    # ── Layer 2: LLM thematic analysis, grounded by NER ──
    prompt = f"""You are an expert story analyst comparing two movie plots.

PLOT A (user's idea):
"{user_plot}"

PLOT B ("{matched_movie}" — an existing film):
"{matched_plot}"

These plots are {similarity_score:.0%} similar according to SBERT + TF-IDF analysis.

{ner_context}
YOUR TASK:
Identify 3 to 5 specific aspects where these two plots are most similar.
Focus on meaningful story elements, not surface-level word matches.
Consider: premise, setting, character archetypes, plot structure, themes,
conflict type, resolution pattern, tone, and genre conventions.

For each aspect, explain clearly what both plots share and why a reader
would notice the overlap.

Respond with ONLY a JSON object:
{{
  "aspects": [
    {{"aspect": "short label (2-4 words)", "explanation": "1-2 sentences explaining the shared element"}},
    ...
  ]
}}"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.3, "num_predict": 2048},
    )

    try:
        data = json.loads(response.message.content)
    except json.JSONDecodeError:
        data = {"aspects": []}

    return {
        "shared_entities": shared,
        "aspects": data.get("aspects", []),
    }


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
        options={"temperature": 0.7, "num_predict": 2048},
    )

    try:
        data = json.loads(response.message.content)
    except json.JSONDecodeError:
        return []
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
