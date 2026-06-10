# QuentinTokenino — The Cinematic Pipeline

**Check if a movie plot is original. If it is, rewrite it in a famous director's style.**

---

## What does this do?

You type a movie plot idea. The system does five things:

1. **Similarity check** — How close is your idea to an existing film? Scored on a spectrum from "Highly Original" to "Near Identical" — not a binary yes/no. (SBERT + TF-IDF + NER entity overlap)
2. **Similarity breakdown** — *What* makes it similar? (spaCy NER + LLM analysis)
3. **Director match** — Which famous director made the closest existing movie? (popularity-weighted ranking)
4. **Style rewrite** — Your plot, rewritten as that director would tell it.
5. **Differentiation** — Suggests how to make your plot more original.

Results stream in progressively — you see the verdict instantly while the LLM works on the rewrite. Everything comes back as clean, structured JSON — and there's a web UI with deterministic quips like *"That's just Star Wars"* or *"Echoes of Inception, but you're building something new."*

---

## Who is this for?

- **Screenwriters** checking if their idea accidentally copies an existing film
- **Development executives** assessing pitch originality before investing
- **Film students** exploring how the same story changes across directorial styles

---

## How it works (step by step)

```
YOU TYPE:  "An astronaut gets stranded on a desert planet
            and has to grow food to survive."
            
                         |
                         v
            
  STEP 1: PLAGIARISM DETECTION
  +--------------------------------------------+
  |  Tool: SBERT + TF-IDF + NER entities       |
  |  What: Compares your plot against 16,000+  |
  |        real movies using three signals:    |
  |        semantic meaning (SBERT), vocabulary |
  |        overlap (TF-IDF), and shared named  |
  |        entities (spaCy NER). Entity matches|
  |        with famous films count more.       |
  |        NOT a cloud API -- runs locally.    |
  |                                            |
  |  Result: "The Martian" (Ridley Scott)      |
  |          Similarity: 0.35                  |
  |          Plagiarism: YES (>= 0.30)         |
  +--------------------------------------------+
                         |
                         v

  STEP 1b: SIMILARITY EXPLANATION
  +--------------------------------------------+
  |  Tool: spaCy NER + Ollama                  |
  |  What: spaCy extracts named entities from  |
  |        both plots (deterministic). Ollama   |
  |        then explains deeper thematic and    |
  |        structural parallels, grounded by    |
  |        the NER findings.                    |
  |                                            |
  |  Result: "Both feature a lone protagonist  |
  |  stranded in a hostile environment who must |
  |  use science to survive."                   |
  +--------------------------------------------+
                         |
                         v

  STEP 2: DIRECTOR ROUTING
  +--------------------------------------------+
  |  "The Martian" was directed by             |
  |  Ridley Scott. He becomes the style target.|
  |                                            |
  |  This is a database lookup, not AI.        |
  +--------------------------------------------+
                         |
                         v

  STEP 3: STYLE REWRITE
  +--------------------------------------------+
  |  Tool: Gemma3 via Ollama (runs locally)    |
  |  What: Rewrites your plot in Ridley        |
  |        Scott's filmmaking style --         |
  |        his pacing, tone, and themes.       |
  |                                            |
  |  THIS IS THE ONLY STEP USING AI.           |
  |  Runs on your machine, no API key needed.  |
  +--------------------------------------------+
                         |
                         v

  STEP 4: VALIDATION
  +--------------------------------------------+
  |  Tool: Pydantic                            |
  |  What: Checks the output has all the       |
  |        right fields and correct types.     |
  |        If anything is wrong, it fails      |
  |        loudly -- no silent errors.         |
  +--------------------------------------------+
                         |
                         v

  OUTPUT (JSON):
  {
    "detected_plagiarism": true,
    "matched_movie": "The Martian",
    "similarity_score": 0.35,
    "assigned_director": "Ridley Scott",
    "rewritten_plot": "...",
    "stylistic_notes": "..."
  }
```

---

## What's in each file?

| File | What it does | Uses AI? |
|---|---|---|
| `app.py` | Flask web server — serves the UI and API | Routes to pipeline |
| `static/index.html` | Web UI — Analyze tab + Evaluation Dashboard | No (frontend only) |
| `pipeline.py` | Main pipeline — runs all 4 steps + similarity/differentiation | Steps 1b, 3 |
| `schema.py` | Defines the output format (6 fields) | No |
| `evaluation.py` | Tests the pipeline with 50 movie plots (varied lengths) | Optional |
| `data_cleaning.ipynb` | Jupyter notebook that merges CMU + MovieSum + IMDB data | No |
| `movies_dataset.csv` | The movie database (16,455 movies, 8,155 directors, popularity scores) | No |
| `download_cache.py` | Downloads pre-computed NLP cache from GitHub Releases | No |
| `requirements.txt` | Python package dependencies | -- |
| `logs/` | Timestamped logs from every run | -- |

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download the spaCy English model
python -m spacy download en_core_web_sm

# 3. Install and start Ollama (https://ollama.com)
# Then pull the model:
ollama pull gemma3

# 4. (Optional) Download pre-computed NLP cache (~52 MB)
#    Skips the ~5 min first-run computation of SBERT embeddings + TF-IDF + NER.
#    If you skip this, the app will compute and cache features on first startup.
python download_cache.py

# 5. The movie database (movies_dataset.csv) is included.
#    To rebuild it from raw sources, run the data_cleaning.ipynb notebook.
```

No API keys needed. Everything runs locally.

---

## Usage

### Web UI (recommended):
```bash
python app.py
# Open http://localhost:8080
```

### Run the pipeline from the command line:
```bash
python pipeline.py
```

### Run the full evaluation (50 test cases, all metrics):
```bash
python evaluation.py
```

### Run evaluation without Ollama (tests SBERT+TF-IDF detection only):
```bash
python evaluation.py --local-only
```

---

## How we measure performance

We test with **50 carefully designed plots** across three tiers and varied lengths (short / medium / long):

| Tier | Count | What | Example |
|---|---|---|---|
| **Blatant plagiarism** | 15 | Near-copies of real films (5 short, 5 medium, 5 long) | "Two hitmen discuss burgers before a hit..." (Pulp Fiction) |
| **Partial overlap** | 15 | Same genre, different story (5 short, 5 medium, 5 long) | "Soldiers in WWII complete a rescue mission" |
| **Fully original** | 20 | No match in the database (mixed lengths) | "A cheese sculptor uncovers a dairy conspiracy" |

Length variety ensures the detector works regardless of whether the user writes one sentence or a full paragraph.

### Three metrics:

| Metric | What it answers |
|---|---|
| **Tool Accuracy** | Does the plagiarism detector get the right answer? |
| **JSON Compliance** | Does the output have valid structure? |
| **Style Adherence** | Does the rewrite sound like the director? (1-5 scale, requires Ollama) |

### Where to find results:

Every run writes:
- **Human-readable log**: `logs/evaluation_<timestamp>.log`
- **Machine-readable JSON**: `logs/eval_results_<timestamp>.json`

Both contain the full trace -- every decision, every score, every match.

---

## Key design decisions

### Why three signals (SBERT + TF-IDF + NER)?

No single signal catches everything:

| Signal | Catches | Misses |
|---|---|---|
| **SBERT** (semantic embeddings) | Paraphrases using different words | Can't tell "same story" from "same genre" |
| **TF-IDF** (vocabulary overlap) | Exact word reuse | Misses synonym-based rewording |
| **NER entity overlap** (spaCy) | Shared character/place names ("Jedi", "Hogwarts") | Doesn't help when plots share no proper nouns |

SBERT and TF-IDF are combined via **geometric mean** (`√(sbert × tfidf)`) — both must be high for a match. NER entity overlap is added as a **bonus** (`+0.20 × entity_recall × popularity_scale`) that only helps, never hurts. The bonus is weighted by IMDB popularity so that matching "Star Wars" entities counts more than matching an obscure film.

| | Three-signal (current) | Cloud embeddings (v0.1) |
|---|---|---|
| **API dependency** | None — runs locally | Required Google API call |
| **Determinism** | Same input = same output, always | Model updates can change results |
| **Speed** | ~10ms per query (pre-computed) | Network round-trip (~500ms) |
| **Length-robust** | Yes — SBERT normalizes meaning | Partially |

### Why popularity-weighted ranking (and entity bonus)?

When two movies score similarly, the more **famous** one ranks higher. IMDB vote counts (log-scaled, normalized to 0–1) also amplify the NER entity bonus: if your plot mentions "lightsabers" and "Jedi," the match against Star Wars (famous) gets a stronger boost than a match against an obscure film with similar terms. Popularity affects both *which* movie is shown and how strongly entity overlap contributes to the score.

### Why 0.30 as the plagiarism threshold?

Calibrated against our 50-case test set (see `evaluation.py`):
- **Blatant plagiarism** scores 0.20 – 0.70 (geometric mean)
- **Partial overlap** scores 0.21 – 0.37
- **Original plots** score 0.15 – 0.32
- **0.30** balances precision and recall for the screenwriter use case
- Conservative alternative: 0.38 (100% precision, lower recall)

### Why Ollama instead of a cloud API?

- **No API key** -- removes the #1 setup friction point
- **No rate limits** -- evaluation suite runs without throttling
- **Privacy** -- plot ideas never leave your machine
- **Free** -- no per-token billing

---

## Architecture (what talks to what)

```
        app.py (Flask)                 static/index.html
       (web server)                     (browser UI)
            |                                |
            +--- /api/analyze ---------------+
            +--- /api/similarity ------------+
            +--- /api/differentiate ---------+
            +--- /api/evaluate --------------+
            |
            v
                    pipeline.py
                   (orchestrator)
                  /   |    |    \
                 /    |    |     \
        schema.py  SBERT  spaCy  ollama
      (validation) TF-IDF (NER) (Gemma3)
              \       |      |      /
               \      |      |     /
              movies_dataset.csv
            (16,455 movies, local)
```

- `app.py` serves the web UI and routes API calls to the pipeline.
- `schema.py` defines the contract -- what the output MUST look like.
- `sentence-transformers` provides SBERT semantic embeddings (all-MiniLM-L6-v2, ~80MB, local).
- `scikit-learn` does TF-IDF vectorization + cosine similarity.
- `spaCy` extracts named entities -- used both for plagiarism scoring (entity recall bonus) and grounded similarity analysis.
- `ollama` does the creativity -- style rewriting and thematic analysis via Gemma3 (local).
- `movies_dataset.csv` is the ground truth -- 16,455 real movies across 8,155 directors.
- `pipeline.py` ties it all together, caches NLP features to disk, and logs every step.

---

## Tech stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| Flask | >= 3.0.0 | Web server + API |
| Pandas | >= 2.2.2 | Data loading |
| sentence-transformers | >= 3.0.0 | SBERT semantic embeddings (all-MiniLM-L6-v2) |
| scikit-learn | >= 1.5.0 | TF-IDF + cosine similarity |
| spaCy | >= 3.7.0 | Named Entity Recognition (NER) |
| Pydantic | >= 2.8.0 | Output validation |
| Ollama | >= 0.6.0 | Local LLM (Gemma3) |
| NumPy | >= 1.26.4 | Array operations |

---

## Directors in the database

**8,155 unique directors** from three merged sources (hand-written + MovieSum + CMU/IMDB).

Top directors by film count: Michael Curtiz (42), Alfred Hitchcock (34),
Clint Eastwood (32), Steven Spielberg (32), Martin Scorsese (28),
Ridley Scott (25), Woody Allen (26), John Ford (24), Frank Capra (22),
plus 8,146 more.
