"""
app.py - Flask web server for the Cinematic Pipeline.

Serves a single-page UI with two tabs:
    1. ANALYZE  - Paste a plot, get plagiarism check + style rewrite + differentiation tips
    2. EVALUATE - Run the 15-case test suite and see all metrics in a dashboard

Endpoints:
    GET  /                  - Serve the HTML UI
    POST /api/analyze       - Run the pipeline on a user plot
    POST /api/differentiate - Get differentiation strategies for a flagged plot
    POST /api/evaluate      - Run the full evaluation suite

Start:
    python app.py
    Then open http://localhost:5000
"""

import json
import traceback

from flask import Flask, send_from_directory, request, jsonify
import pandas as pd

from pipeline import (
    run_pipeline,
    detect_plagiarism,
    suggest_differentiation,
    setup_logger,
    OLLAMA_MODEL,
)
from evaluation import run_evaluation, EVAL_CASES
from schema import PLAGIARISM_THRESHOLD

app = Flask(__name__, static_folder="static")

# Pre-load the database once at startup
CSV_PATH = "movies_dataset.csv"
DF = pd.read_csv(CSV_PATH)


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
        # Run TF-IDF detection first (fast, for top matches)
        detection = detect_plagiarism(plot, DF)

        # Run full pipeline (includes LLM rewrite)
        result = run_pipeline(plot, CSV_PATH)

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


@app.route("/api/differentiate", methods=["POST"])
def differentiate():
    """
    Get differentiation strategies for a plot that's too similar.

    Expects JSON: {"plot": "...", "matched_movie": "...",
                   "assigned_director": "...", "similarity_score": float}
    Returns JSON: {"strategies": [...]}
    """
    data = request.get_json() or {}
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
    Run the full 15-case evaluation suite.

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
    print(f"\n  Open http://localhost:5000\n")
    app.run(debug=True, port=5000)
