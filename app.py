"""
app.py - Flask web server for the Cinematic Pipeline.

Serves a single-page UI with two tabs:
    1. ANALYZE  - Paste a plot, get plagiarism check + style rewrite + differentiation tips
    2. EVALUATE - Run the 50-case test suite and see all metrics in a dashboard

Endpoints:
    GET  /                  - Serve the HTML UI
    POST /api/analyze       - Run the pipeline on a user plot
    POST /api/similarity    - Explain what's most similar (spaCy NER + LLM)
    POST /api/differentiate - Get differentiation strategies for a flagged plot
    POST /api/evaluate      - Run the full evaluation suite

Start:
    python app.py
    Then open http://localhost:8080
"""

import json
import traceback

from flask import Flask, Response, send_from_directory, request, jsonify, stream_with_context
import pandas as pd

from pipeline import (
    run_pipeline,
    detect_plagiarism,
    init_nlp,
    explain_similarity,
    suggest_differentiation,
    rewrite_in_director_style,
    validate_output,
    OLLAMA_MODEL,
)
from evaluation import run_evaluation, EVAL_CASES
from schema import PLAGIARISM_THRESHOLD

app = Flask(__name__, static_folder="static")

# Pre-load the database and fit all NLP features once at startup
CSV_PATH = "movies_dataset.csv"
DF = pd.read_csv(CSV_PATH)
init_nlp(DF)


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main UI."""
    return send_from_directory("static", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Run the full pipeline on a user-submitted plot.

    Expects JSON: {"plot": "your plot text here"}
    Returns JSON: full MovieAnalysis + detection metadata
    """
    data = request.get_json()
    plot = (data or {}).get("plot", "").strip()

    if not plot:
        return jsonify({"error": "Please enter a plot description."}), 400

    try:
        # Run TF-IDF detection once, then pass result into pipeline
        detection = detect_plagiarism(plot, DF)

        # Run full pipeline, reusing cached DF and detection result
        result = run_pipeline(plot, df=DF, detection=detection)

        return jsonify({
            "success": True,
            "result": result.model_dump(),
            "top_matches": detection["top_matches"],
            "threshold": PLAGIARISM_THRESHOLD,
            "model": OLLAMA_MODEL,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze-stream", methods=["POST"])
def analyze_stream():
    """
    Streaming version of /api/analyze using Server-Sent Events.

    Sends results progressively as each pipeline step completes:
        1. detection  — instant (~10ms): score, category, quip, top matches
        2. rewrite    — LLM call (~15-30s): director-style rewrite + notes
        3. similarity — NER + LLM (~15-30s): shared entities + thematic aspects
        4. complete   — signals end of stream

    The UI renders each section as its data arrives, so the user sees
    the verdict immediately instead of waiting 30-60s for everything.
    """
    data = request.get_json()
    plot = (data or {}).get("plot", "").strip()

    if not plot:
        return jsonify({"error": "Please enter a plot description."}), 400

    def sse(event_type, payload):
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    def generate():
        try:
            # ── Step 1: Detection (instant, ~10ms) ──
            detection = detect_plagiarism(plot, DF)

            yield sse("detection", {
                "detection": detection,
                "threshold": PLAGIARISM_THRESHOLD,
            })

            # ── Step 2: Style Rewrite (LLM, ~15-30s) ──
            raw_json = rewrite_in_director_style(
                user_plot=plot,
                matched_movie=detection["matched_movie"],
                assigned_director=detection["assigned_director"],
                similarity_score=detection["similarity_score"],
                detected_plagiarism=detection["detected_plagiarism"],
            )
            result = validate_output(raw_json)

            yield sse("rewrite", {
                "result": result.model_dump(),
                "model": OLLAMA_MODEL,
            })

            # ── Step 3: Similarity Explanation (NER + LLM, ~15-30s) ──
            movie_row = DF[DF["title"] == detection["matched_movie"]]
            if not movie_row.empty:
                matched_plot = movie_row.iloc[0]["plot"]
                if isinstance(matched_plot, str) and matched_plot.strip():
                    sim_result = explain_similarity(
                        user_plot=plot,
                        matched_movie=detection["matched_movie"],
                        matched_plot=matched_plot,
                        similarity_score=detection["similarity_score"],
                    )
                    yield sse("similarity", sim_result)

            yield sse("complete", {})

        except Exception as e:
            traceback.print_exc()
            yield sse("error", {"error": str(e)})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/similarity", methods=["POST"])
def similarity():
    """
    Explain what's most similar between the user's plot and the closest match.

    Uses two layers:
        1. spaCy NER to find shared named entities (deterministic)
        2. Ollama to identify deeper thematic/structural parallels

    Expects JSON: {"plot": "...", "matched_movie": "...", "similarity_score": float}
    Returns JSON: {"shared_entities": [...], "aspects": [...]}
    """
    data = request.get_json() or {}

    required = ["plot", "matched_movie", "similarity_score"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        # Look up the matched movie's plot from the database
        movie_row = DF[DF["title"] == data["matched_movie"]]
        if movie_row.empty:
            return jsonify({"error": f"Movie not found: {data['matched_movie']}"}), 404
        matched_plot = movie_row.iloc[0]["plot"]
        if not isinstance(matched_plot, str) or not matched_plot.strip():
            return jsonify({"error": f"No plot text available for: {data['matched_movie']}"}), 404

        result = explain_similarity(
            user_plot=data["plot"],
            matched_movie=data["matched_movie"],
            matched_plot=matched_plot,
            similarity_score=data["similarity_score"],
        )
        return jsonify({"success": True, **result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/differentiate", methods=["POST"])
def differentiate():
    """
    Get differentiation strategies for a plot that's too similar.

    Expects JSON: {"plot": "...", "matched_movie": "...",
                   "assigned_director": "...", "similarity_score": float}
    Returns JSON: {"strategies": [...]}
    """
    data = request.get_json() or {}

    # Validate required fields
    required = ["plot", "matched_movie", "assigned_director", "similarity_score"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        strategies = suggest_differentiation(
            user_plot=data["plot"],
            matched_movie=data["matched_movie"],
            assigned_director=data["assigned_director"],
            similarity_score=data["similarity_score"],
        )
        return jsonify({"success": True, "strategies": strategies})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """
    Run the full 50-case evaluation suite.

    Expects JSON: {"local_only": bool}  (optional, default false)
    Returns JSON: full evaluation report with all metrics
    """
    data = request.get_json() or {}
    local_only = data.get("local_only", False)

    try:
        report = run_evaluation(
            csv_path=CSV_PATH,
            run_llm=not local_only,
        )
        return jsonify({"success": True, "report": report})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/info", methods=["GET"])
def info():
    """Return system info for the UI header."""
    return jsonify({
        "model": OLLAMA_MODEL,
        "threshold": PLAGIARISM_THRESHOLD,
        "database_size": len(DF),
        "directors": sorted(DF["director"].unique().tolist()),
        "eval_cases": len(EVAL_CASES),
    })


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n  Cinematic Pipeline UI")
    print(f"  Model: {OLLAMA_MODEL} (via Ollama)")
    print(f"  Database: {len(DF)} movies, {DF['director'].nunique()} directors")
    print(f"  Threshold: {PLAGIARISM_THRESHOLD}")
    print(f"\n  Open http://localhost:8080\n")
    app.run(debug=True, port=8080)
