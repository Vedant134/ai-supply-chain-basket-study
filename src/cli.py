"""Command-line entry point for the frozen-data research workflow."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import download_snapshot, load_snapshot
from .study import run_study, save_study


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="AI versus infrastructure basket study")
    parser.add_argument("command", choices=("fetch", "run", "all"))
    parser.add_argument("--refresh", action="store_true", help="replace the frozen Yahoo snapshot")
    parser.add_argument("--end", help="exclusive YYYY-MM-DD end for a new download")
    args = parser.parse_args()

    if args.command == "fetch" or (args.command == "all" and (args.refresh or not (DATA_DIR / "prices_adjusted.csv").exists())):
        manifest = download_snapshot(DATA_DIR, end=args.end)
        print(f"Saved {manifest['complete_sessions']} common price sessions to {DATA_DIR}")
        if args.command == "fetch":
            return

    if args.command in ("run", "all"):
        manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
        snapshot = DATA_DIR / "prices_adjusted.csv"
        if sha256(snapshot.read_bytes()).hexdigest() != manifest["clean_sha256"]:
            raise ValueError("Frozen price snapshot changed since download; restore or refresh it.")
        prices = load_snapshot(DATA_DIR)
        study = run_study(prices)
        save_study(study, OUTPUT_DIR, manifest)
        print(f"Saved results to {OUTPUT_DIR / 'REPORT.md'}")
        print("Baseline total return:", f"{study.metrics['baseline']['total_return']:.2%}")
        print("ML-filtered total return:", f"{study.metrics['ml']['total_return']:.2%}")


if __name__ == "__main__":
    main()
