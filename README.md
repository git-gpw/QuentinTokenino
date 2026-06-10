# QuentinTokenino — The Cinematic Pipeline

**Check if a movie plot is original. If it is, rewrite it in a famous director's style.**

---

## What does this do?

You type a movie plot idea. The system tells you:

1. **How similar is it?** — Scored on a spectrum from "Highly Original" to "Near Identical," with a one-liner like *"That's just Star Wars"* or *"Echoes of Inception, but you're building something new."*
2. **What makes it similar?** — Which specific elements (characters, places, themes) overlap with existing films.
3. **Who made the closest match?** — Which famous director's movie yours most resembles.
4. **What would it sound like in their style?** — Your plot, rewritten as that director would tell it.
5. **How to make it more original** — Specific suggestions to create distance from existing films.

Results appear as they're ready — you see the verdict instantly while the AI works on the rewrite in the background. Everything runs locally on your machine. No API keys, no cloud calls, no data leaving your computer.

---

## Who is this for?

- **Screenwriters** checking if their idea accidentally copies an existing film
- **Development executives** assessing pitch originality before investing
- **Film students** exploring how the same story changes across directorial styles

---

## How it works

```
YOU TYPE:  "An astronaut gets stranded on a desert planet
            and has to grow food to survive."

                         |
                         v

  STEP 1: SIMILARITY SCORING (instant, ~10ms)
  +--------------------------------------------+
  |  Compares your plot against 16,455 real     |
  |  movies using three independent signals:    |
  |                                             |
  |  1. Meaning:  Does it tell the same story?  |
  |  2. Wording:  Does it use the same words?   |
  |  3. Names:    Does it share character or     |
  |               place names? (Rare names like  |
  |               "Jedi" count more than common  |
  |               ones like "American.")         |
  |                                             |
  |  Result: "The Martian" (Ridley Scott)       |
  |          35% similar — "Strongly Similar"   |
  +--------------------------------------------+
                         |
        results start appearing in the UI
                         |
                         v

  STEP 2: STYLE REWRITE (~15-30s)
  +--------------------------------------------+
  |  A local AI (Gemma3 via Ollama) rewrites    |
  |  your plot in Ridley Scott's filmmaking     |
  |  style — his pacing, tone, and themes.      |
  |                                             |
  |  This is the only step using AI.            |
  |  Runs on your machine, no API key needed.   |
  +--------------------------------------------+
                         |
                         v

  STEP 3: SIMILARITY BREAKDOWN (~15-30s)
  +--------------------------------------------+
  |  Identifies the specific shared elements:   |
  |  character types, settings, plot structure,  |
  |  themes. Grounded by named entity analysis  |
  |  (deterministic) + AI thematic analysis.    |
  +--------------------------------------------+
                         |
                         v

  OUTPUT:
  {
    "similarity_score": 0.35,
    "category": "Strongly Similar",
    "quip": "Basically The Martian with a fresh coat of paint.",
    "matched_movie": "The Martian",
    "assigned_director": "Ridley Scott",
    "shared_elements": ["mars", "nasa"],
    "rewritten_plot": "...",
    "stylistic_notes": "..."
  }
```

---

## The similarity spectrum

Instead of a binary "plagiarism yes/no," scores map to five categories:

| Category | Score Range | What it means | Example quip |
|---|---|---|---|
| **Highly Original** | 0 – 15% | No meaningful overlap with existing films | *"Fresh off the imagination — no doppelgängers in sight."* |
| **Minor Similarities** | 15 – 25% | Faint resemblance, likely coincidence | *"A hint of Se7en — probably coincidence."* |
| **Notable Similarities** | 25 – 35% | Some shared DNA, but the story diverges | *"Echoes of Inception, but you're building something new."* |
| **Strongly Similar** | 35 – 50% | Clear parallels to an existing film | *"Basically The Martian with a fresh coat of paint."* |
| **Near Identical** | 50%+ | Essentially the same story | *"That's just Star Wars."* |

Quips are deterministic — same input always produces the same quip. They use the matched movie name and, when available, the specific shared elements (e.g. *"The Jedi give it away — very Star Wars."*).

---

## How scoring works (plain English)

The score combines three independent checks. Think of it like three different people reading your plot:

**Person 1 — The Meaning Reader.**
*"Does this tell the same story as an existing movie?"*
Uses sentence embeddings (SBERT) to understand what your plot *means*, not just what words it uses. Catches paraphrases — if you describe Inception's plot without using any of its words, this still detects it.

**Person 2 — The Word Counter.**
*"Does this use the same vocabulary as an existing movie?"*
Uses TF-IDF to check for shared keywords. Distinguishes "this is a heist movie" (genre overlap, common words) from "this is about dream extraction" (specific vocabulary match).

**Person 3 — The Name Checker.**
*"Does this mention the same characters, places, or things?"*
Uses Named Entity Recognition (spaCy) to find shared proper nouns. Important: not all name matches are equal. Each name is weighted by how **rare** it is:

| Name | Appears in... | Weight | Why |
|---|---|---|---|
| "Jedi" | 5 movies | **0.83** (strong signal) | Very distinctive — almost certainly refers to Star Wars |
| "Mars" | 35 movies | **0.70** | Fairly specific |
| "Italian" | 500+ movies | **0.50** (weak signal) | Too common to mean much |
| "American" | 2,000+ movies | **0.33** (near-noise) | Appears everywhere |

This weighting uses the same concept as search engines: **Inverse Document Frequency (IDF)**. Words that appear in many documents are less informative than words that appear in few. We apply this twice — once for how rare the name is as an extracted entity, and once for how rare it is as a plain word — so a name has to be genuinely unusual in both senses to carry weight.

**Combining the three:**
- Persons 1 and 2 must *both* agree for a base match (geometric mean — both scores must be high)
- Person 3 adds a bonus on top, scaled by how famous the matched movie is (matching Star Wars counts more than matching an obscure film)

---

## What's in each file?

| File | What it does | Uses AI? |
|---|---|---|
| `app.py` | Web server — serves the UI and streaming API | Routes to pipeline |
| `static/index.html` | Web UI — Analyze tab + Evaluation Dashboard | No (frontend only) |
| `pipeline.py` | Main pipeline — scoring, rewriting, similarity analysis | Rewrite + similarity |
| `schema.py` | Defines the output format and similarity categories | No |
| `evaluation.py` | Tests the pipeline with 50 movie plots (varied lengths) | Optional |
| `data_cleaning.ipynb` | Jupyter notebook that builds the movie database from raw sources | No |
| `movies_dataset.csv` | The movie database (16,455 movies, 8,155 directors, popularity scores) | No |
| `download_cache.py` | Downloads pre-computed features from GitHub Releases | No |
| `requirements.txt` | Python package dependencies | — |
| `logs/` | Timestamped logs from every run | — |

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

# 4. (Optional) Download pre-computed features (~52 MB)
#    Skips the ~5 min first-run computation.
#    If you skip this, the app computes and caches on first startup.
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

### Command line:
```bash
python pipeline.py
```

### Run the full evaluation (50 test cases):
```bash
python evaluation.py              # Full (detection + LLM rewrite)
python evaluation.py --local-only  # Detection only (no Ollama needed)
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

### Metrics:

| Metric | What it answers |
|---|---|
| **Tool Accuracy** | Does the detector get the right answer? |
| **JSON Compliance** | Does the output have valid structure? |
| **Style Adherence** | Does the rewrite sound like the director? (1–5 scale, requires Ollama) |

---

## Architecture

```
        app.py (Flask)                 static/index.html
       (web server)                     (browser UI)
            |                                |
            +--- /api/analyze-stream --------+  (SSE streaming)
            +--- /api/analyze ---------------+  (batch, for eval)
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

- **`app.py`** serves the web UI and routes API calls. The streaming endpoint (`/api/analyze-stream`) uses Server-Sent Events to push results progressively.
- **`pipeline.py`** orchestrates scoring, caches pre-computed features to disk, and logs every step.
- **`schema.py`** defines the output contract and the five similarity categories.
- **SBERT** (sentence-transformers) provides semantic embeddings — understands *meaning*.
- **TF-IDF** (scikit-learn) provides vocabulary overlap — catches shared *words*.
- **spaCy** extracts named entities — catches shared *names*, weighted by rarity (IDF).
- **Ollama** (Gemma3) does the creative work — style rewriting and thematic analysis. Runs locally.
- **`movies_dataset.csv`** is the ground truth — 16,455 real movies across 8,155 directors.

---

## Tech stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| Flask | >= 3.0.0 | Web server + streaming API |
| Pandas | >= 2.2.2 | Data loading |
| sentence-transformers | >= 3.0.0 | Semantic embeddings (all-MiniLM-L6-v2) |
| scikit-learn | >= 1.5.0 | TF-IDF + cosine similarity |
| spaCy | >= 3.7.0 | Named Entity Recognition |
| Pydantic | >= 2.8.0 | Output validation |
| Ollama | >= 0.6.0 | Local LLM (Gemma3) |
| NumPy | >= 1.26.4 | Array operations |

---

## Design decisions

### Why not just use one similarity method?

No single method catches everything:
- **Meaning-based** (SBERT) catches paraphrases but can't tell "same story" from "same genre"
- **Word-based** (TF-IDF) catches exact reuse but misses synonym-based rewording
- **Name-based** (NER) catches shared proper nouns but doesn't help when plots share no names

By requiring multiple signals to agree, we reduce false positives. The geometric mean of SBERT and TF-IDF means *both* must be high — a movie in the same genre won't trigger unless it also shares specific vocabulary.

### Why weight entity matches by rarity?

Without weighting, "Italian" matching would count as much as "Jedi" matching. But "Italian" appears in 500+ movie plots — it's not a meaningful signal. We use Inverse Document Frequency (IDF), the same concept search engines use, to automatically downweight common terms and amplify rare ones.

### Why a spectrum instead of yes/no?

A binary "plagiarized or not" decision is both too harsh and too simple. A 25% similarity to The Matrix means something very different from 60%. The five-category spectrum lets users make their own judgment about how much similarity is acceptable for their use case.

### Why popularity-weighted ranking?

When two movies score similarly, the more famous one ranks higher. If your plot resembles both an obscure 1940s B-movie and The Matrix, you probably want to know about The Matrix. IMDB vote counts (log-scaled) serve as the fame signal.

### Why run everything locally?

- **No API key** — removes the #1 setup friction point
- **No rate limits** — evaluation suite runs without throttling
- **Privacy** — plot ideas never leave your machine
- **Free** — no per-token billing
- **Deterministic** — same input always produces the same score

---

## Directors in the database

**8,155 unique directors** from three merged sources (hand-written + MovieSum + CMU/IMDB).

Top directors by film count: Michael Curtiz (42), Alfred Hitchcock (34),
Clint Eastwood (32), Steven Spielberg (32), Martin Scorsese (28),
Ridley Scott (25), Woody Allen (26), John Ford (24), Frank Capra (22),
plus 8,146 more.
