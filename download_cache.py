"""
download_cache.py — Fetch pre-computed NLP cache from GitHub Releases.

Run this once after cloning to skip the ~5 minute first-run computation.
If you don't run this, the app will compute and cache features on first startup.

Usage:
    python download_cache.py
"""

import os
import sys
import urllib.request

CACHE_FILE = "nlp_cache.pkl"
RELEASE_URL = "https://github.com/git-gpw/QuentinTokenino/releases/latest/download/nlp_cache.pkl"


def download():
    if os.path.exists(CACHE_FILE):
        size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
        print(f"  {CACHE_FILE} already exists ({size_mb:.1f} MB). Delete it first to re-download.")
        return

    print(f"  Downloading {CACHE_FILE} from GitHub Releases...")
    print(f"  URL: {RELEASE_URL}")
    print(f"  This is ~52 MB and may take a minute.\n")

    try:
        urllib.request.urlretrieve(RELEASE_URL, CACHE_FILE, _progress)
        print(f"\n  Done! Saved to {CACHE_FILE}")
        size_mb = os.path.getsize(CACHE_FILE) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print("  The app will compute features on first startup instead (~5 min).")
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        sys.exit(1)


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  [{pct:3d}%] {mb:.1f} / {total_mb:.1f} MB")
        sys.stdout.flush()


if __name__ == "__main__":
    download()
