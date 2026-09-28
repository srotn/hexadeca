"""Generate organized JSON and PNG outputs for one Hexadeca record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis import analyze_game
from analysis.plot_game_summary import render_summary


def analyze_record(record: Path, output_root: Path) -> tuple[Path, Path]:
    """Analyze one record into ``json/`` and ``png/`` output folders."""

    json_path = output_root / "json" / f"{record.stem}-features.json"
    png_path = output_root / "png" / f"{record.stem}-summary.png"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(record.read_text(encoding="utf-8"))
    board_size = len(payload.get("final_matrix", []))
    result = analyze_game(record, board_size=board_size)
    json_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    render_summary(record, png_path)
    return json_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis"),
        help="Root containing the json/ and png/ output folders",
    )
    arguments = parser.parse_args()
    json_path, png_path = analyze_record(arguments.record, arguments.output_root)
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
