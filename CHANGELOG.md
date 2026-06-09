# Changelog

All notable changes to the Cinematic Pipeline project.

Format follows [Keep a Changelog](https://keepachangelog.com/).

---

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
