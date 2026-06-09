"""
evaluation.py - Evaluation suite for the Cinematic Pipeline.

WHAT THIS DOES (plain English):
================================
Runs 15 movie plots through the pipeline and measures how well it works.

We test three things:
    1. TOOL ACCURACY    - Does the plagiarism detector get the right answer?
    2. JSON COMPLIANCE  - Does the output have valid structure?
    3. STYLE ADHERENCE  - Does the rewrite actually sound like the director?

Everything is logged to logs/evaluation_<timestamp>.log and a JSON results
file so you can inspect exactly what happened after the fact.


TEST SET DESIGN:
================
    5 BLATANT PLAGIARISM  - Near-paraphrases of real films. Should be caught.
    5 PARTIAL OVERLAP     - Same genre but different story. Should NOT be caught.
    5 FULLY ORIGINAL      - No database match at all. Should NOT be caught.


METRICS EXPLAINED:
==================
    ACCURACY   = (correct answers) / (total cases)
                 "How often did it get the right yes/no?"

    PRECISION  = (true positives) / (true positives + false positives)
                 "When it said plagiarism, was it right?"

    RECALL     = (true positives) / (true positives + false negatives)
                 "Of all real plagiarism cases, how many did it catch?"

    COMPLIANCE = (valid JSON outputs) / (total cases)
                 "How often did the LLM return properly structured output?"

    STYLE MEAN = average score (1-5) from an independent LLM judge
                 "How well do rewrites capture the director's voice?"
"""

import os
import json
import time
import logging
from datetime import datetime

import pandas as pd

from pipeline import detect_plagiarism, run_pipeline, setup_logger, OLLAMA_MODEL
from schema import PLAGIARISM_THRESHOLD


# -----------------------------------------------------------------------
# 15 Test Cases: ground truth labels for plagiarism detection
# -----------------------------------------------------------------------

EVAL_CASES = [
    # === TIER 1: BLATANT PLAGIARISM (expected: True) ====================
    {
        "id": "PLAG-01",
        "tier": "blatant_plagiarism",
        "user_plot": (
            "Two mob hitmen discuss cheeseburgers and divine intervention "
            "before carrying out a hit for their gangster boss in Los Angeles."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Pulp Fiction",
    },
    {
        "id": "PLAG-02",
        "tier": "blatant_plagiarism",
        "user_plot": (
            "After uncovering a mysterious artifact beneath the lunar surface, "
            "a spacecraft crew embarks on a mission toward Jupiter guided by a "
            "sentient supercomputer that begins to malfunction."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of 2001: A Space Odyssey",
    },
    {
        "id": "PLAG-03",
        "tier": "blatant_plagiarism",
        "user_plot": (
            "A German bounty hunter frees a slave and together they rescue "
            "the slave's wife from a brutal Mississippi plantation owner."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Django Unchained",
    },
    {
        "id": "PLAG-04",
        "tier": "blatant_plagiarism",
        "user_plot": (
            "A thief who steals secrets from people's dreams is hired to do "
            "the reverse - plant an idea deep inside a target's subconscious mind."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Inception",
    },
    {
        "id": "PLAG-05",
        "tier": "blatant_plagiarism",
        "user_plot": (
            "Two homicide detectives hunt a methodical serial killer who "
            "stages gruesome murders based on the seven deadly sins."
        ),
        "expected_plagiarism": True,
        "notes": "Near-paraphrase of Se7en",
    },

    # === TIER 2: PARTIAL OVERLAP (expected: False) ======================
    {
        "id": "PART-01",
        "tier": "partial_overlap",
        "user_plot": (
            "A group of soldiers must survive behind enemy lines in World War II "
            "to complete a dangerous rescue mission in France."
        ),
        "expected_plagiarism": False,
        "notes": "Generic WWII rescue - shares tropes with Saving Private Ryan",
    },
    {
        "id": "PART-02",
        "tier": "partial_overlap",
        "user_plot": (
            "A genius mathematician struggles with mental illness while working "
            "on classified government projects during the Cold War."
        ),
        "expected_plagiarism": False,
        "notes": "Resembles A Beautiful Mind (not in DB), may partially match Oppenheimer",
    },
    {
        "id": "PART-03",
        "tier": "partial_overlap",
        "user_plot": (
            "A boxer past his prime gets one last shot at the championship, "
            "training alone in the rough streets of a decaying city."
        ),
        "expected_plagiarism": False,
        "notes": "Boxing tropes - may partially match Raging Bull or Million Dollar Baby",
    },
    {
        "id": "PART-04",
        "tier": "partial_overlap",
        "user_plot": (
            "A detective investigates a string of disappearances on a remote "
            "island where nothing is what it seems and reality bends."
        ),
        "expected_plagiarism": False,
        "notes": "Mystery-island tropes - may partially match Shutter Island",
    },
    {
        "id": "PART-05",
        "tier": "partial_overlap",
        "user_plot": (
            "In a dystopian future, a lone officer is tasked with hunting down "
            "rogue artificial beings who have escaped their intended purpose."
        ),
        "expected_plagiarism": False,
        "notes": "Android-hunting tropes - similar to Blade Runner but reworded",
    },

    # === TIER 3: ORIGINAL (expected: False) =============================
    {
        "id": "ORIG-01",
        "tier": "original",
        "user_plot": (
            "A retired librarian discovers that the books in her basement are "
            "rewriting themselves overnight, each one predicting a local "
            "disaster 24 hours before it happens."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original concept",
    },
    {
        "id": "ORIG-02",
        "tier": "original",
        "user_plot": (
            "A competitive cheese sculptor in rural Vermont uncovers a "
            "conspiracy among dairy farmers to replace all artisan cheese "
            "with synthetic substitutes."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original, absurdist concept",
    },
    {
        "id": "ORIG-03",
        "tier": "original",
        "user_plot": (
            "Twin sisters separated at birth - one raised by monks in Tibet, "
            "the other by a jazz band in New Orleans - accidentally meet at "
            "an airport baggage claim and swap lives."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original concept",
    },
    {
        "id": "ORIG-04",
        "tier": "original",
        "user_plot": (
            "A sentient traffic light in Tokyo gains consciousness and begins "
            "subtly rerouting cars to prevent accidents, drawing the attention "
            "of a suspicious city engineer."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original, high-concept",
    },
    {
        "id": "ORIG-05",
        "tier": "original",
        "user_plot": (
            "An aging perfumer in Marseille attempts to recreate the exact "
            "scent of a thunderstorm she experienced as a child, believing "
            "it holds the key to a suppressed memory."
        ),
        "expected_plagiarism": False,
        "notes": "Fully original concept",
    },
]


# -----------------------------------------------------------------------
# METRIC 1: Tool Accuracy
# -----------------------------------------------------------------------

def compute_tool_accuracy(results: list[dict], log: logging.Logger) -> dict:
    """
    Compare the TF-IDF tool's yes/no plagiarism calls against ground truth.

    Plain English:
        - True Positive (TP):  Tool said plagiarism, and it WAS plagiarism.
        - False Positive (FP): Tool said plagiarism, but it WASN'T.
        - True Negative (TN):  Tool said original, and it WAS original.
        - False Negative (FN): Tool said original, but it WAS plagiarism.
    """
    tp = fp = tn = fn = 0

    for r in results:
        pred = r.get("predicted_plagiarism")
        true = r["expected_plagiarism"]
        if pred is None:
            continue
        if pred and true:
            tp += 1
        elif pred and not true:
            fp += 1
        elif not pred and not true:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_scored": total,
    }

    log.info("")
    log.info("METRIC 1: Tool Accuracy (Plagiarism Detection)")
    log.info(f"  Accuracy:    {accuracy:.1%}  ({tp + tn} of {total} correct)")
    log.info(f"  Precision:   {precision:.1%}  (when it says plagiarism, is it right?)")
    log.info(f"  Recall:      {recall:.1%}  (of real plagiarism, how much did it catch?)")
    log.info(f"  Breakdown:   TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    return metrics


# -----------------------------------------------------------------------
# METRIC 2: JSON Schema Compliance
# -----------------------------------------------------------------------

def compute_schema_compliance(results: list[dict], log: logging.Logger) -> dict:
    """How many LLM outputs passed Pydantic validation?"""
    valid   = [r for r in results if r["schema_valid"]]
    invalid = [r for r in results if not r["schema_valid"] and r.get("llm_attempted")]
    total_attempted = len(valid) + len(invalid)
    rate = len(valid) / total_attempted if total_attempted else 0

    metrics = {
        "passed": len(valid),
        "failed": len(invalid),
        "total_attempted": total_attempted,
        "compliance_rate": round(rate, 4),
    }

    log.info("")
    log.info("METRIC 2: JSON Schema Compliance")
    log.info(f"  Passed:      {len(valid)} / {total_attempted}  ({rate:.1%})")
    if invalid:
        log.info(f"  Failed IDs:  {[r['id'] for r in invalid]}")

    return metrics


# -----------------------------------------------------------------------
# METRIC 3: Style Adherence (LLM-as-a-Judge)
# -----------------------------------------------------------------------

def judge_style_adherence(
    user_plot: str,
    rewritten_plot: str,
    director: str,
) -> tuple[int, str]:
    """
    Ask an independent LLM call to score how well the rewrite
    captures the target director's filmmaking style.

    Runs locally via Ollama — same model, separate call, acting as
    an independent judge rather than the author.

    Scoring:
        1 = No resemblance to the director's style
        2 = Slight resemblance
        3 = Moderate resemblance
        4 = Strong resemblance
        5 = Unmistakably in the director's style

    Returns:
        (score, justification_string)
    """
    prompt = f"""You are a film studies professor grading student work.

ORIGINAL PLOT:
"{user_plot}"

REWRITTEN PLOT:
"{rewritten_plot}"

TARGET DIRECTOR: {director}

Score the rewrite on how well it captures {director}'s filmmaking style.
Respond with ONLY a JSON object: {{"score": <1-5>, "justification": "<one sentence>"}}

Rubric:
  1 = No resemblance to the director's style
  2 = Slight resemblance
  3 = Moderate resemblance
  4 = Strong resemblance
  5 = Unmistakably in the director's style"""

    import ollama

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0.0},
    )

    data = json.loads(response.message.content)
    return int(data.get("score", 0)), data.get("justification", "")


def compute_style_scores(results: list[dict], log: logging.Logger) -> dict:
    """Aggregate the LLM-as-a-Judge style scores."""
    scores = [r["style_score"] for r in results if r["style_score"] is not None]

    if not scores:
        log.info("")
        log.info("METRIC 3: Style Adherence - no scores available (LLM calls may have been skipped)")
        return {"mean_score": None, "count": 0}

    mean_s = sum(scores) / len(scores)
    metrics = {
        "mean_score": round(mean_s, 2),
        "min_score": min(scores),
        "max_score": max(scores),
        "count": len(scores),
    }

    log.info("")
    log.info("METRIC 3: Style Adherence (LLM-as-a-Judge, 1 to 5)")
    log.info(f"  Mean score:  {mean_s:.2f} / 5.0")
    log.info(f"  Range:       {min(scores)} - {max(scores)}  (n={len(scores)})")

    return metrics


# -----------------------------------------------------------------------
# Main evaluation runner
# -----------------------------------------------------------------------

def run_evaluation(
    csv_path: str = "movies_dataset.csv",
    run_llm: bool = True,
):
    """
    Run all 15 test cases, compute all metrics, log everything.

    Args:
        csv_path: Path to movie database CSV.
        run_llm:  If False, only runs Step 1 (TF-IDF) and skips LLM calls.
                  Useful for testing detection accuracy without an API key.
    """
    log = setup_logger("evaluation")

    log.info("=" * 70)
    log.info("CINEMATIC PIPELINE - EVALUATION SUITE")
    log.info(f"Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Test cases:           {len(EVAL_CASES)}")
    log.info(f"Plagiarism threshold: {PLAGIARISM_THRESHOLD}")
    log.info(f"LLM calls enabled:   {run_llm}")
    log.info("=" * 70)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Movie database not found: {csv_path}")
    df = pd.read_csv(csv_path)

    results = []

    for i, case in enumerate(EVAL_CASES):
        log.info("")
        log.info(f"--- Case {i + 1}/15: {case['id']} [{case['tier']}] ---")
        log.info(f"  Plot:     \"{case['user_plot'][:80]}...\"")
        log.info(f"  Expected: {'PLAGIARISM' if case['expected_plagiarism'] else 'ORIGINAL'}")

        result = {
            "id": case["id"],
            "tier": case["tier"],
            "expected_plagiarism": case["expected_plagiarism"],
            "predicted_plagiarism": None,
            "similarity_score": None,
            "matched_movie": None,
            "assigned_director": None,
            "schema_valid": False,
            "llm_attempted": False,
            "style_score": None,
            "style_justification": None,
            "error": None,
        }

        try:
            # ---- STEP 1: TF-IDF detection (always runs, no API needed) ----
            detection = detect_plagiarism(case["user_plot"], df)
            result["predicted_plagiarism"] = detection["detected_plagiarism"]
            result["similarity_score"] = detection["similarity_score"]
            result["matched_movie"] = detection["matched_movie"]
            result["assigned_director"] = detection["assigned_director"]

            correct = detection["detected_plagiarism"] == case["expected_plagiarism"]
            symbol = "CORRECT" if correct else "WRONG"

            log.info(f"  Match:    \"{detection['matched_movie']}\" (score: {detection['similarity_score']})")
            log.info(f"  Predicted: {'PLAGIARISM' if detection['detected_plagiarism'] else 'ORIGINAL'}  [{symbol}]")

            # ---- STEPS 2-4: LLM rewrite + validation (optional) ----
            if run_llm:
                result["llm_attempted"] = True
                pipeline_result = run_pipeline(case["user_plot"], csv_path, log=log)
                result["schema_valid"] = True

                # ---- METRIC 3: LLM-as-a-Judge ----
                score, justification = judge_style_adherence(
                    case["user_plot"],
                    pipeline_result.rewritten_plot,
                    pipeline_result.assigned_director,
                )
                result["style_score"] = score
                result["style_justification"] = justification
                log.info(f"  Style:    {score}/5 - {justification}")

        except Exception as e:
            result["error"] = str(e)
            log.error(f"  ERROR: {e}")

        results.append(result)
        if run_llm:
            time.sleep(2)  # rate-limit courtesy

    # ---- SUMMARY ----
    log.info("")
    log.info("=" * 70)
    log.info("EVALUATION RESULTS SUMMARY")
    log.info("=" * 70)

    tool_metrics   = compute_tool_accuracy(results, log)
    schema_metrics = compute_schema_compliance(results, log)
    style_metrics  = compute_style_scores(results, log)

    # Per-case table
    log.info("")
    header = f"{'ID':<10} {'Tier':<22} {'Expect':<10} {'Predict':<10} {'Score':<8} {'Movie':<28} {'JSON':<6} {'Style':<6}"
    log.info(header)
    log.info("-" * len(header))
    for r in results:
        log.info(
            f"{r['id']:<10} "
            f"{r['tier']:<22} "
            f"{'PLAG' if r['expected_plagiarism'] else 'ORIG':<10} "
            f"{'PLAG' if r['predicted_plagiarism'] else 'ORIG' if r['predicted_plagiarism'] is not None else '?':<10} "
            f"{r['similarity_score'] or 0:<8.4f} "
            f"{(r['matched_movie'] or '-')[:26]:<28} "
            f"{'PASS' if r['schema_valid'] else '-':<6} "
            f"{r['style_score'] if r['style_score'] is not None else '-'!s:<6}"
        )

    # Save raw results JSON
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"logs/eval_results_{ts}.json"

    report = {
        "run_timestamp": ts,
        "config": {
            "plagiarism_threshold": PLAGIARISM_THRESHOLD,
            "llm_enabled": run_llm,
            "total_cases": len(EVAL_CASES),
        },
        "metrics": {
            "tool_accuracy": tool_metrics,
            "schema_compliance": schema_metrics,
            "style_adherence": style_metrics,
        },
        "cases": results,
    }

    with open(results_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"\nFull results saved to: {results_path}")
    log.info(f"Log saved to: {getattr(log, 'log_path', 'unknown')}")

    return report


# -----------------------------------------------------------------------
# Standalone execution
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the Cinematic Pipeline")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Skip LLM calls, only test TF-IDF plagiarism detection accuracy",
    )
    args = parser.parse_args()

    print("\nRunning Cinematic Pipeline Evaluation Suite...\n")
    report = run_evaluation(run_llm=not args.local_only)

    print("\n--- QUICK SUMMARY ---")
    m = report["metrics"]
    print(f"Tool Accuracy:     {m['tool_accuracy']['accuracy']:.1%}")
    print(f"Schema Compliance: {m['schema_compliance']['compliance_rate']:.1%}")
    if m["style_adherence"].get("mean_score"):
        print(f"Style Adherence:   {m['style_adherence']['mean_score']:.1f} / 5.0")
    print(f"\nDetailed log: logs/")
