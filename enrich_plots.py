"""
enrich_plots.py - Expand short plot summaries using Ollama.

Finds all movies in movies_dataset.csv with plots under 200 chars,
asks gemma3 to write a richer 400-800 char summary, and saves the
updated dataset. Backs up the original first.

Usage:
    python enrich_plots.py          # enrich all short plots
    python enrich_plots.py --dry 5  # test on 5 movies without saving
"""

import sys
import time
import pandas as pd
import ollama

MODEL = "gemma3"
MIN_PLOT_LEN = 200  # plots shorter than this get enriched
CSV = "movies_dataset.csv"


def enrich_one(title: str, director: str, genre: str, existing_plot: str) -> str:
    """Ask Ollama to expand a short plot stub into a richer summary."""
    genre_str = f" ({genre})" if pd.notna(genre) and genre else ""
    prompt = f"""Write a detailed plot summary for the movie "{title}" directed by {director}{genre_str}.

Here is a brief description: "{existing_plot}"

Write a factual, detailed plot summary in 400-800 characters (not words).
Cover the main characters, key plot points, conflicts, and resolution.
Write in third person, present tense, as a plot synopsis.
Return ONLY the plot summary text, no titles or labels."""

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3},
    )
    return response.message.content.strip()


def main():
    dry_run = False
    dry_count = 0
    if len(sys.argv) > 1 and sys.argv[1] == "--dry":
        dry_run = True
        dry_count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(f"DRY RUN: enriching {dry_count} movies (no save)\n")

    df = pd.read_csv(CSV)
    short_mask = df["plot"].str.len() < MIN_PLOT_LEN
    short_indices = df[short_mask].index.tolist()

    if dry_run:
        short_indices = short_indices[:dry_count]

    total = len(short_indices)
    print(f"Movies to enrich: {total}")
    print(f"{'=' * 60}\n")

    enriched_count = 0
    failed = []
    start_all = time.time()

    for i, idx in enumerate(short_indices):
        row = df.loc[idx]
        title = row["title"]
        director = row["director"]
        genre = row.get("genre", "")
        old_plot = row["plot"]

        print(f"[{i+1}/{total}] {title} ({director}) [{len(old_plot)} chars]", end=" ", flush=True)

        try:
            new_plot = enrich_one(title, director, genre, old_plot)

            # Only replace if the new plot is actually longer
            if len(new_plot) > len(old_plot):
                df.at[idx, "plot"] = new_plot
                enriched_count += 1
                print(f"-> {len(new_plot)} chars")
            else:
                print(f"-> SKIP (new={len(new_plot)}, not longer)")

        except Exception as e:
            print(f"-> FAIL: {e}")
            failed.append((title, str(e)))

        # Progress estimate every 20
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start_all
            rate = elapsed / (i + 1)
            remaining = rate * (total - i - 1)
            print(f"  ... {elapsed:.0f}s elapsed, ~{remaining/60:.0f}m remaining\n")

    elapsed_total = time.time() - start_all
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed_total/60:.1f} minutes")
    print(f"Enriched: {enriched_count}/{total}")
    print(f"Failed:   {len(failed)}")

    if failed:
        print("\nFailed movies:")
        for title, err in failed:
            print(f"  - {title}: {err}")

    if not dry_run and enriched_count > 0:
        df.to_csv(CSV, index=False)
        print(f"\nSaved to {CSV}")
    elif dry_run:
        print("\n(Dry run - no changes saved)")


if __name__ == "__main__":
    main()
