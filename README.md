# QuentinTokenino — The Cinematic Pipeline

**Check if a movie plot is original. If it is, rewrite it in a famous director's style.**

---

## What does this do?

You type a movie plot idea. The system does three things:

1. **Plagiarism check** — Is your idea too close to an existing film?
2. **Director match** — Which famous director made the closest existing movie?
3. **Style rewrite** — Your plot, rewritten as that director would tell it.
4. **Differentiation** — If flagged, suggests how to make your plot more original.

Everything comes back as clean, structured JSON — and there's a web UI.

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
  |  Tool: TF-IDF (scikit-learn)               |
  |  What: Compares your plot against 170+     |
  |        real movies using word importance    |
  |        scores -- NOT AI, just math.        |
  |                                            |
  |  Result: "The Martian" (Ridley Scott)      |
  |          Similarity: 0.35                  |
  |          Plagiarism: YES (>= 0.30)         |
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
| `pipeline.py` | Main pipeline — runs all 4 steps | Step 3 only |
| `schema.py` | Defines the output format (6 fields) | No |
| `evaluation.py` | Tests the pipeline with 15 movie plots | Optional |
| `generate_dataset.py` | Creates the 170-movie database CSV | No |
| `movies_dataset.csv` | The movie database (title, director, genre, plot) | No |
| `requirements.txt` | Python package dependencies | -- |
| `logs/` | Timestamped logs from every run | -- |

### Legacy files (from v0.1, no longer used):
| File | Why it's here |
|---|---|
| `agent.py` | Old pipeline using neural embeddings. Replaced by `pipeline.py`. |
| `generate_embeddings.py` | Generated embedding vectors. Not needed with TF-IDF. |

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install and start Ollama (https://ollama.com)
# Then pull the model:
ollama pull gemma3

# 3. Generate the movie database (if movies_dataset.csv doesn't exist)
python generate_dataset.py
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

### Run the full evaluation (15 test cases, all metrics):
```bash
python evaluation.py
```

### Run evaluation without Ollama (tests plagiarism detection only):
```bash
python evaluation.py --local-only
```

---

## How we measure performance

We test with **15 carefully designed plots** in three tiers:

| Tier | Count | What | Example |
|---|---|---|---|
| **Blatant plagiarism** | 5 | Near-copies of real films | "Two hitmen discuss burgers before a hit..." (Pulp Fiction) |
| **Partial overlap** | 5 | Same genre, different story | "Soldiers in WWII complete a rescue mission" |
| **Fully original** | 5 | No match in the database | "A cheese sculptor uncovers a dairy conspiracy" |

### Three metrics:

| Metric | What it answers | Current score |
|---|---|---|
| **Tool Accuracy** | Does the plagiarism detector get the right answer? | 100% (15/15) |
| **JSON Compliance** | Does the output have valid structure? | Requires Ollama |
| **Style Adherence** | Does the rewrite sound like the director? (1-5 scale) | Requires Ollama |

### Where to find results:

Every run writes:
- **Human-readable log**: `logs/evaluation_<timestamp>.log`
- **Machine-readable JSON**: `logs/eval_results_<timestamp>.json`

Both contain the full trace -- every decision, every score, every match.

---

## Key design decisions

### Why TF-IDF instead of neural embeddings?

| | TF-IDF (current) | Neural Embeddings (old) |
|---|---|---|
| **Explainability** | "These specific words overlap" | "The vectors are close" (black box) |
| **API dependency** | None -- runs locally | Required Google API call |
| **Determinism** | Same input = same output, always | Model updates can change results |
| **Speed** | Instant (~10ms) | Network round-trip (~500ms) |
| **Weakness** | Misses synonym-based similarity | Catches semantic similarity |

We chose TF-IDF because the plagiarism step should be **explainable and deterministic**. If someone asks "why did you flag my plot?", we can point to exact word overlaps -- not just "the AI thought so."

### Why 0.30 as the plagiarism threshold?

Calibrated against our test set:
- **Plagiarism cases** score between 0.32 and 0.81
- **Non-plagiarism cases** score between 0.06 and 0.26
- **0.30** sits in the gap, giving 100% accuracy with zero false positives

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
            +--- /api/differentiate ---------+
            +--- /api/evaluate --------------+
            |
            v
                    pipeline.py
                   (orchestrator)
                    /    |     \
                   /     |      \
        schema.py    scikit-learn    ollama
     (output format)  (TF-IDF)    (Gemma3 LLM)
              \          |           /
               \         |          /
              movies_dataset.csv
              (196 movies, local)
```

- `app.py` serves the web UI and routes API calls to the pipeline.
- `schema.py` defines the contract -- what the output MUST look like.
- `scikit-learn` does the math -- TF-IDF vectorization + cosine similarity.
- `ollama` does the creativity -- style rewriting via Gemma3 (local).
- `movies_dataset.csv` is the ground truth -- 196 real movies across 16 directors.
- `pipeline.py` ties it all together and logs every step.

---

## Tech stack

| Tool | Version | Role |
|---|---|---|
| Python | 3.10+ | Runtime |
| Flask | >= 3.0.0 | Web server + API |
| Pandas | >= 2.2.2 | Data loading |
| scikit-learn | >= 1.5.0 | TF-IDF + cosine similarity |
| Pydantic | >= 2.8.0 | Output validation |
| Ollama | >= 0.6.0 | Local LLM (Gemma3) |
| NumPy | >= 1.26.4 | Array operations |

---

## Directors in the database

Quentin Tarantino, Christopher Nolan, Wes Anderson, Martin Scorsese,
Stanley Kubrick, Steven Spielberg, Ridley Scott, David Fincher,
Denis Villeneuve, Alfred Hitchcock, James Cameron, Francis Ford Coppola,
Clint Eastwood, Joel Coen, Peter Jackson, Guillermo del Toro
