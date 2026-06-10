"""
schema.py - Structured output definitions for the Cinematic Pipeline.

Defines the Pydantic models that enforce JSON structure on all pipeline outputs.
The LLM's response MUST conform to these schemas or validation fails.

CHANGE LOG:
    v2 - Fixed threshold inconsistency. Single source of truth: PLAGIARISM_THRESHOLD = 0.30
    v3 - Switched LLM backend from Google Gemini to local Ollama (gemma3).
"""

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for the plagiarism decision boundary.
# Score = geometric mean of SBERT + TF-IDF, plus NER entity overlap bonus.
#   final = sqrt(sbert * tfidf) + 0.20 * entity_recall * pop_scale
#
# Calibrated against our 50-case test set (see evaluation.py).
# Three-signal scores on the 16,455-movie enriched dataset:
#   - Blatant paraphrases score 0.20 - 0.70+
#   - Partial overlap scores    0.21 - 0.37
#   - Original plots score      0.15 - 0.32
# 0.30 balances precision and recall for the screenwriter use case.
# Conservative alternative: 0.38 (100% precision, lower recall).
# NOTE: threshold may need recalibration after adding entity bonus.
# -----------------------------------------------------------------------
PLAGIARISM_THRESHOLD = 0.30


# -----------------------------------------------------------------------
# Likert-style similarity categories — replaces binary yes/no in the UI.
# The threshold above is still used internally (evaluation, logging),
# but users see a spectrum from "Highly Original" to "Near Identical".
# -----------------------------------------------------------------------
SIMILARITY_CATEGORIES = [
    {"id": "highly_original",      "max": 0.15, "label": "Highly Original",      "color": "#2ecc71"},
    {"id": "minor_similarities",   "max": 0.25, "label": "Minor Similarities",   "color": "#1abc9c"},
    {"id": "notable_similarities", "max": 0.35, "label": "Notable Similarities", "color": "#f39c12"},
    {"id": "strongly_similar",     "max": 0.50, "label": "Strongly Similar",     "color": "#e67e22"},
    {"id": "near_identical",       "max": 1.50, "label": "Near Identical",       "color": "#e94560"},
]


def classify_similarity(score: float) -> dict:
    """Map a similarity score to a Likert-style category."""
    for cat in SIMILARITY_CATEGORIES:
        if score < cat["max"]:
            return dict(cat)  # return a copy
    return dict(SIMILARITY_CATEGORIES[-1])


class MovieAnalysis(BaseModel):
    """
    The validated output of the cinematic pipeline.

    Every field is enforced by Pydantic - if the LLM returns bad JSON,
    validation fails loudly rather than passing garbage downstream.

    Fields:
        detected_plagiarism : bool   - Did the SBERT+TF-IDF tool flag this as too similar?
        matched_movie       : str    - Which existing movie was the closest match?
        similarity_score    : float  - How similar? (0.0 = nothing in common, 1.0 = identical)
        assigned_director   : str    - Whose style should the rewrite follow?
        rewritten_plot      : str    - The user's plot, rewritten in that director's voice
        stylistic_notes     : str    - Plain-English explanation of what changed and why
    """

    detected_plagiarism: bool = Field(
        description=(
            f"True if SBERT semantic similarity >= {PLAGIARISM_THRESHOLD}, "
            "False otherwise. Set by the detection tool, NOT the LLM."
        )
    )
    matched_movie: str = Field(
        description="Title of the closest matching movie from the database"
    )
    similarity_score: float = Field(
        description="SBERT+TF-IDF geometric mean similarity between 0.0 and 1.0"
    )
    assigned_director: str = Field(
        description="Director of the matched movie, used as the style target for rewriting"
    )
    rewritten_plot: str = Field(
        description="The user's plot rewritten in the assigned director's filmmaking style"
    )
    stylistic_notes: str = Field(
        description="Plain-English explanation of what stylistic choices were made and why"
    )
