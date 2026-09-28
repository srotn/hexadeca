"""Command-line entry point for Hexadeca feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.features import analyze_game


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Hexadeca D/G/I/S/B features from a game JSON"
    )
    parser.add_argument("record", type=Path)
    parser.add_argument(
        "--allow-nonterminal",
        action="store_true",
        help="Do not require the move sequence to be terminal",
    )
    arguments = parser.parse_args()
    result = analyze_game(
        arguments.record,
        require_terminal=not arguments.allow_nonterminal,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover - command-line convenience
    main()
