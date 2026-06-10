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
# Any TF-IDF cosine similarity score >= this value = plagiarism.
#
# Why 0.30?
#   Calibrated against our 15-case test set (see evaluation.py).
#   TF-IDF cosine scores on the 16,455-movie dataset:
#     - Blatant paraphrases score 0.31 - 0.90  (all above threshold)
#     - Partial overlap scores  0.14 - 0.29  (all below threshold)
#     - Original plots score    0.12 - 0.18  (all below threshold)
#   0.30 achieves 100% accuracy (15/15) with zero false positives.
# -----------------------------------------------------------------------
PLAGIARISM_THRESHOLD = 0.30


class MovieAnalysis(BaseModel):
    """
    The validated output of the cinematic pipeline.

    Every field is enforced by Pydantic - if the LLM returns bad JSON,
    validation fails loudly rather than passing garbage downstream.

    Fields:
        detected_plagiarism : bool   - Did the TF-IDF tool flag this as too similar?
        matched_movie       : str    - Which existing movie was the closest match?
        similarity_score    : float  - How similar? (0.0 = nothing in common, 1.0 = identical)
        assigned_director   : str    - Whose style should the rewrite follow?
        rewritten_plot      : str    - The user's plot, rewritten in that director's voice
        stylistic_notes     : str    - Plain-English explanation of what changed and why
    """

    detected_plagiarism: bool = Field(
        description=(
            f"True if TF-IDF cosine similarity >= {PLAGIARISM_THRESHOLD} (0.30), "
            "False otherwise. Set by the detection tool, NOT the LLM."
        )
    )
    matched_movie: str = Field(
        description="Title of the closest matching movie from the database"
    )
    similarity_score: float = Field(
        description="TF-IDF cosine similarity between 0.0 and 1.0"
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
