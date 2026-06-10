# Changelog

All notable changes to the Cinematic Pipeline project.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

## [0.9.0] - 2026-06-10

### Added
- **Likert-style similarity categories** replace binary plagiarism yes/no.
  Five levels: Highly Original, Minor Similarities, Notable Similarities,
  Strongly Similar, Near Identical. Each has a distinct color in the UI.
- **Deterministic quips** — witty one-liners based on score category, movie
  name, and shared entity elements. Same input always produces the same quip.
  Examples: "That's just Star Wars." / "Echoes of Inception, but you're
  building something new."
- **Server-Sent Events (SSE) streaming** endpoint `/api/analyze-stream`.
  Results arrive progressively: detection (instant) → rewrite (~15-30s) →
  similarity (~15-30s). No more waiting 60s for everything at once.
- **Progressive UI rendering** — each result section fades in as its data
  arrives. Loading indicator dynamically updates with current task name
  (e.g. "Rewriting in Ridley Scott's style...").
- `shared_elements` field in detection output — the named entities the user's
  plot shares with the best-matched movie.
- `download_cache.py` — downloads pre-computed NLP cache (~52 MB) from GitHub
  Releases to skip the ~5 min first-run computation.
- **GitHub Release v0.9.0** with `nlp_cache.pkl` as a binary asset.

### Fixed
- **NER noise entity filtering** — CARDINAL ("two", "one"), ORDINAL, DATE,
  TIME, QUANTITY, MONEY, PERCENT entity types are now excluded from plagiarism
  scoring. These matched everywhere and inflated scores. Example: "Two
  astronauts on a mission to Mars" previously matched Apollo 13 at 37% due to
  "two", "one", "earth" — now correctly matches Red Planet at 46% via "mars",
  "earth".
- Cache versioning (`_NLP_CACHE_VERSION = 2`) forces automatic re-extraction
  of entity sets when upgrading from v1 cache, while preserving SBERT/TF-IDF.

### Changed
- Verdict card uses category-based colors and labels instead of binary
  "Plagiarism Detected" / "Original Plot".
- Score bar thresholds updated to align with Likert categories.
- Search history shows category labels instead of PLAG/ORIG badges.
- Rewrite and stylistic notes merged into a single card for cleaner layout.


## [0.8.0] - 2026-06-10

### Added
- **NER entity overlap** as third plagiarism detection signal. spaCy extracts
  named entities from the user's plot and computes recall against each DB
  movie's pre-computed entity set. Applied as additive bonus (0.20 * recall)
  — only helps, never hurts existing scores.
- **Popularity-weighted entity bonus**: entity overlap with famous movies
  (high IMDB numVotes) is amplified; obscure matches are dampened. Scale
  factor: 0.5 (obscure) to 1.0 (famous).
- Entity sets cached to `nlp_cache.pkl` alongside SBERT and TF-IDF features.
  Old caches trigger a one-time backfill (~3 min).

### Fixed
- **Keyword description blind spot**: short plots mentioning distinctive proper
  nouns (e.g. "Jedi", "lightsabers") now correctly match their source movies.
  Previously scored 0.19 for Star Wars; now scores 0.39+ and flags plagiarism.
- False positive for "cheese sculptor" plot dropped from 0.32 to 0.26 because
  the incidental entity match ("Vermont") is against an obscure film.
- "undefined" text in Most Similar Aspects UI when LLM returns variant key
  names (`name` instead of `aspect`, `description` instead of `explanation`).


## [0.7.0] - 2026-06-10

### Summary
Replaced TF-IDF-only plagiarism detection with dual-signal SBERT + TF-IDF
(geometric mean), added IMDB popularity-weighted ranking, expanded evaluation
from 15 to 50 test cases, and enriched 872 short plot summaries.

### Added
- **SBERT semantic embeddings** — `sentence-transformers` (all-MiniLM-L6-v2, ~80MB)
  encodes all plots into 384-dim vectors for semantic similarity. Combined with
  TF-IDF via geometric mean: `sqrt(sbert * tfidf)` requires both signals high.
- **Popularity scoring** — IMDB `numVotes` (log-scaled, normalized 0–1) added to
  `movies_dataset.csv`. Famous movies rank higher when scores are close.
  Formula: `ranking_score = combined * (1 + 0.15 * popularity)`.
- **Disk caching** — `nlp_cache.pkl` stores pre-computed SBERT embeddings and
  TF-IDF matrix with SHA-256 hash validation. First run ~3 min, subsequent ~2s.
  Atomic writes (temp file + rename) prevent corruption.
- **50 evaluation test cases** in `evaluation.py` — 15 blatant plagiarism
  (5 short / 5 medium / 5 long), 15 partial overlap, 20 original. Tests
  length robustness.
- Cell 5b in `data_cleaning.ipynb` — loads `title.ratings.tsv.gz`, joins to
  merged dataset via title+director, computes popularity column.

### Changed
- **Detection method**: TF-IDF cosine → geometric mean of SBERT + TF-IDF.
  SBERT catches paraphrases, TF-IDF distinguishes genre overlap from plagiarism.
- **Threshold calibration**: 0.30 revalidated against 50-case dual-signal scores.
  Blatant plagiarism: 0.20–0.70, partial overlap: 0.21–0.37, original: 0.15–0.32.
- `schema.py` field descriptions updated from "TF-IDF" to "SBERT+TF-IDF".
- `requirements.txt`: added `sentence-transformers>=3.0.0`.
- `README.md`: rewritten to document dual-signal detection, popularity ranking,
  and 50-case evaluation.

### Fixed
- `evaluation.py` never called `init_nlp()` — each of 50 test cases re-encoded
  all 16k plots from scratch (~150 min instead of ~30s).
- TF-IDF fallback path included user plot in fitting corpus, producing different
  scores than the cached path (could flip plagiarism decision at threshold boundary).
- Cache writes were not atomic — interrupted writes left corrupted `nlp_cache.pkl`
  with no warning, causing silent slow restarts.
- Stale "15-case" docstring in `/api/evaluate` endpoint.
- Stale "TF-IDF text analysis" in `explain_similarity()` LLM prompt.


## [0.6.0] - 2026-06-10

### Summary
Enriched 872 short movie plots, UI polish, and accumulated bug fixes
from code review (PR #1).

### Added
- Enriched 872 short plot summaries (< 200 chars) to ~1200 chars via
  Ollama LLM. Improves TF-IDF and SBERT signal quality.

### Fixed
- Alert() dialogs replaced with inline error banners in the UI.
- Stale flow indicators after errors.
- NaN/empty plot guard in `/api/similarity` lookup.
- Hardcoded `/15` in evaluation log line.
- Logger file descriptor leak in `run_pipeline`.
- Stale accuracy comment in `schema.py`.
- `num_predict` cap and `JSONDecodeError` handling for Ollama calls.
- Passed `df=` and `detection=` to `run_pipeline` in evaluation loop
  to avoid redundant CSV reads and duplicate detection.
- Pre-fit TF-IDF vectorizer at startup (114x faster per request).

### Removed
- Dead legacy files: `agent.py`, `generate_embeddings.py`, `generate_dataset.py`.

### Changed
- `.gitignore`: exclude backup CSVs, enrich script, broader `.env` pattern.
- Search history and rotating placeholder examples in UI dashboard.
- Accept placeholder suggestion with right arrow or Tab key.


## [0.5.0] - 2026-06-09

### Summary
Added spaCy NER for deterministic similarity analysis, expanded movie database
from 196 to 16,455 films, and added a "Most Similar Aspects" UI panel.

### Added
- **spaCy NER integration** — `extract_entities()` and `find_shared_entities()`
  in `pipeline.py`. Deterministic named entity extraction provides a grounded,
  explainable layer before the LLM interprets thematic similarity.
- `explain_similarity()` in `pipeline.py` — Two-layer analysis combining spaCy
  NER (shared entities) + Ollama (thematic/structural parallels).
- `/api/similarity` endpoint in `app.py`.
- "Most Similar Aspects" card in the web UI — auto-fetches after analysis,
  shows shared entity chips (from NER) and thematic aspects (from LLM).
- `data_cleaning.ipynb` — Jupyter notebook that merges CMU Movie Summary Corpus,
  HuggingFace MovieSum, and IMDB datasets into the final database.

### Changed
- **Movie database**: expanded from 196 hand-written movies to 16,455 movies
  across 8,155 directors (merged from 3 sources via IMDB director lookup).
- `requirements.txt`: added `spacy>=3.7.0` and `datasets>=2.20.0`.
- Architecture diagram updated (spaCy as a fourth processing layer).

### Verified
- All 15 evaluation cases still pass at 100% accuracy with the larger dataset.


## [0.3.0] - 2026-06-09

### Summary
Added a web UI, switched to local Ollama (no API keys), and added
LLM-powered differentiation strategies.

### Added
- `app.py` — Flask web server with 4 API endpoints.
- `static/index.html` — Single-page UI with two tabs:
  - **Analyze**: paste a plot, see plagiarism verdict, score bar,
    top matches, director-style rewrite, and differentiation tips.
  - **Evaluate**: run the 15-case test suite with metrics dashboard.
- `suggest_differentiation()` in `pipeline.py` — when a plot is flagged,
  suggests 3-5 specific changes to make it more original.
- Animated process flow indicator showing Steps 1-4 in real time.

### Changed
- **LLM backend**: switched from Google Gemini API to local Ollama (gemma3).
  No API key needed, no rate limits, fully offline.
- `requirements.txt`: replaced `google-genai` with `ollama` and added `flask`.

### Removed
- All Google API key handling (`GOOGLE_API_KEY`, `GEMINI_API_KEY` env vars).


## [0.2.0] - 2026-06-09

### Summary
Complete pipeline rewrite. Replaced opaque neural embeddings with explainable
TF-IDF, fixed contradictory thresholds, built a real evaluation suite, added
logging throughout.

### Added
- `pipeline.py` — New main pipeline replacing `agent.py`. Clean 4-step
  architecture: TF-IDF detection → director routing → LLM rewrite → validation.
- `evaluation.py` — 15-case test suite (5 plagiarism, 5 partial, 5 original)
  with three computed metrics: tool accuracy, schema compliance, style adherence.
- `CHANGELOG.md` — This file.
- Timestamped logging to `logs/` directory for full run transparency.
- `--local-only` flag for evaluation without an API key (tests TF-IDF only).
- LLM-as-a-Judge scoring (style adherence 1–5) via independent Gemini call.
- Top-5 match reporting in pipeline logs for decision transparency.

### Changed
- **schema.py** — Single plagiarism threshold (0.30) as `PLAGIARISM_THRESHOLD`
  constant. Was 0.45 in schema.py and 0.75 in agent.py (contradictory).
  Calibrated against the 15-case test set: 0.30 achieves 100% accuracy with
  clear separation between plagiarism scores (0.32–0.81) and non-plagiarism (0.06–0.26).
- **Similarity method** — TF-IDF + cosine (scikit-learn, local, deterministic)
  replaces Google text-embedding-004 (API-dependent, opaque).
- **LLM prompt** — Now receives hard facts from TF-IDF with explicit
  instructions not to override the plagiarism decision.
- Pipeline functions accept optional logger for caller-controlled logging.

### Deprecated
- `agent.py` — Superseded by `pipeline.py`. Kept for reference.
- `generate_embeddings.py` — No longer needed (TF-IDF doesn't use embeddings).

### Fixed
- Threshold inconsistency between schema.py (0.45) and agent.py (0.75).
- Schema field description claimed "TF-IDF" but code used neural embeddings.

### Security
- `generate_embeddings.py` line 6 contains a hardcoded API key. This file
  is now deprecated but the key should be rotated if it was ever valid.


## [0.1.0] - Initial

### Added
- `agent.py` — Original pipeline using Google text-embedding-004.
- `schema.py` — Pydantic output schema.
- `generate_dataset.py` — 170+ movie dataset.
- `generate_embeddings.py` — Embedding generation script.
- `evaluation.py` — 3-case evaluation stub.
- `requirements.txt` — Dependencies.
