"""Render the README figures from the archived data beside this script.

Requires Matplotlib. Run from any directory:
    python docs/assets/render_showcase.py

Only presentation is regenerated: no training, self-play, or Elo fitting runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE = "#2563EB", "#F97316"
BLUE_LIGHT, ORANGE_LIGHT = "#BFDBFE", "#FED7AA"
INK, MUTED, GRID = "#17243B", "#596B83", "#DCE4EF"
STAGES = ("#475569", "#F97316", "#8B5CF6", "#0F766E", BLUE)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.titleweight": "bold",
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    """Keep the native aspect ratio and a fixed canvas, with no tight cropping."""
    fig.savefig(ROOT / name, dpi=160)
    plt.close(fig)
    print(name)


def heading(fig, title, subtitle):
    fig.text(0.055, 0.955, title, fontsize=24, weight="bold", va="top")
    fig.text(0.055, 0.905, subtitle, fontsize=12, color=MUTED, va="top")


def board(ax, record, title):
    matrix = record["final_matrix"]
    # Occupied cells retain a white background so stones remain distinct.
    cells = [[1 if v == 2 else 2 if v == -2 else 0 for v in row] for row in matrix]
    ax.imshow(
        cells, cmap=ListedColormap(["white", BLUE_LIGHT, ORANGE_LIGHT]), vmin=0, vmax=2
    )
    for value, color in ((1, BLUE), (-1, ORANGE)):
        points = [
            (r, c)
            for r, row in enumerate(matrix)
            for c, v in enumerate(row)
            if v == value
        ]
        ax.scatter(
            [c for r, c in points],
            [r for r, c in points],
            s=48,
            c=color,
            edgecolors="white",
            linewidths=1.1,
            zorder=3,
        )
    ax.set_xticks(range(0, 16, 4))
    ax.set_yticks(range(0, 16, 4))
    ax.set_xticks([i - 0.5 for i in range(17)], minor=True)
    ax.set_yticks([i - 0.5 for i in range(17)], minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Column", fontsize=10)
    ax.set_ylabel("Row", fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_aspect("equal")


def board_legend(fig, y):
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=BLUE,
                label="Black stones (blue)",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=ORANGE,
                label="White stones (orange)",
            ),
            Patch(facecolor=BLUE_LIGHT, label="Black territory"),
            Patch(facecolor=ORANGE_LIGHT, label="White territory"),
            Patch(facecolor="white", edgecolor=GRID, label="Neutral squares"),
        ],
        loc="center",
        bbox_to_anchor=(0.5, y),
        ncol=5,
        frameon=False,
        fontsize=10,
    )


def chart_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def comparison(data):
    games = data["games"]
    fig = plt.figure(figsize=(18, 10), facecolor="white")
    heading(
        fig,
        "Five checkpoints. Five self-play games.",
        "Hexadeca 16 × 16  |  1,600 simulations per move  |  No root noise  |  One game per checkpoint",
    )
    for i, game in enumerate(games):
        meta = game["record"]["metadata"]
        ax = fig.add_axes((0.055 + i * 0.187, 0.53, 0.155, 0.28))
        board(
            ax,
            game["record"],
            f"Iteration {meta['checkpoint_id'].split('-')[1]}\n{meta['plies']} plies  |  {meta['black_score']} : {meta['white_score']}",
        )
        if i:
            ax.set_ylabel("")
    board_legend(fig, 0.472)

    # Titles and legends have their own rows, outside the plotting areas.
    fig.text(0.055, 0.405, "Terminal position descriptors", fontsize=15, weight="bold")
    fig.text(0.605, 0.405, "Game length and score margin", fontsize=15, weight="bold")
    ax = fig.add_axes((0.055, 0.135, 0.485, 0.195))
    for i, game in enumerate(games):
        values = [game["features"][key] for key in ("D", "I", "S0", "B0")]
        ax.bar(
            [x + (i - 2) * 0.15 for x in range(4)], values, width=0.15, color=STAGES[i]
        )
    ax.set_xticks(
        range(4),
        ["D\nDivergence", "I\nIntrusion", "S0\nOverall symmetry", "B0\nEdge affinity"],
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Feature value")
    chart_style(ax)
    fig.legend(
        handles=[
            Patch(
                color=color,
                label=game["record"]["metadata"]["checkpoint_id"].split("-")[1],
            )
            for color, game in zip(STAGES, games, strict=True)
        ],
        loc="center",
        bbox_to_anchor=(0.297, 0.365),
        ncol=5,
        frameon=False,
        fontsize=10,
    )

    ax = fig.add_axes((0.605, 0.135, 0.355, 0.195))
    for offset, key, color, label in (
        (-0.18, "plies", BLUE, "Plies"),
        (0.18, "margin", ORANGE, "Black − White score"),
    ):
        values = [
            g["record"]["metadata"]["plies"]
            if key == "plies"
            else g["record"]["metadata"]["black_score"]
            - g["record"]["metadata"]["white_score"]
            for g in games
        ]
        bars = ax.bar(
            [i + offset for i in range(5)], values, width=0.32, color=color, label=label
        )
        ax.bar_label(bars, padding=3, fontsize=9, color=INK)
    ax.set_xticks(
        range(5),
        [g["record"]["metadata"]["checkpoint_id"].split("-")[1] for g in games],
    )
    ax.set_ylim(0, 50)
    ax.set_ylabel("Plies / points")
    chart_style(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.782, 0.365),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.055,
        0.035,
        "Each checkpoint plays both colors. Scores above are Black : White. Single-game illustrations do not establish statistical significance.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "hexadeca-five-checkpoint-comparison.png")


def heatmap(
    fig,
    rect,
    values,
    title,
    cmap,
    *,
    window=False,
    minimum=0,
    maximum=1,
    extend="neither",
):
    ax = fig.add_axes(rect)
    image = ax.imshow(
        values, cmap=cmap, vmin=minimum, vmax=maximum, interpolation="nearest"
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("Window start column" if window else "Column", fontsize=10)
    ax.set_ylabel("Window start row" if window else "Row", fontsize=10)
    ax.set_xticks(range(0, len(values[0]), 2 if window else 4))
    ax.set_yticks(range(0, len(values), 2 if window else 4))
    ax.tick_params(labelsize=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035, extend=extend)
    colorbar.ax.tick_params(labelsize=9)
    ax.set_aspect("equal")


def terminal_analysis(data):
    game = data["games"][-1]
    features = game["features"]
    detail = data["detail_iteration_004520"]
    fig = plt.figure(figsize=(16, 12), facecolor="white")
    heading(
        fig,
        "A closer look at the final position",
        "Iteration 004520  |  37 plies  |  Black 129 : White 127  |  Five geometric feature families",
    )
    board(fig.add_axes((0.055, 0.545, 0.265, 0.275)), game["record"], "Final board")
    heatmap(
        fig,
        (0.385, 0.545, 0.245, 0.275),
        detail["gridness"]["sparse"],
        "Local gridness · sparse lattice\n6 × 6 windows",
        "Blues",
        window=True,
    )
    heatmap(
        fig,
        (0.71, 0.545, 0.245, 0.275),
        detail["gridness"]["dense"],
        "Local gridness · dense lattice\n4 × 4 windows",
        "Blues",
        window=True,
    )

    ax = fig.add_axes((0.055, 0.115, 0.275, 0.27))
    ax.set_axis_off()
    ax.text(
        0,
        1.06,
        "Geometric descriptors",
        fontsize=13,
        weight="bold",
        transform=ax.transAxes,
    )
    rows = [
        ("D · Divergence", "D"),
        ("I · Intrusion", "I"),
        ("S0 · Overall symmetry", "S0"),
        ("S1 · Black symmetry", "S1"),
        ("S2 · White symmetry", "S2"),
        ("B0 · Overall edge affinity", "B0"),
        ("B1 · Black edge affinity", "B1"),
        ("B2 · White edge affinity", "B2"),
    ]
    for i, (label, key) in enumerate(rows):
        y = 0.93 - i * 0.069
        ax.text(0, y, label, fontsize=10, transform=ax.transAxes)
        ax.text(
            1,
            y,
            f"{features[key]:.6f}",
            ha="right",
            fontsize=10,
            family="DejaVu Sans Mono",
            transform=ax.transAxes,
        )
    ax.text(
        0, 0.325, "Gridness (G)", fontsize=10, weight="bold", transform=ax.transAxes
    )
    table = ax.table(
        cellText=[
            [label, *[f"{v:.3f}" for v in features[key]]]
            for label, key in (("All", "G0"), ("Black", "G1"), ("White", "G2"))
        ],
        colLabels=["", "g1", "g2", "g3", "g4"],
        colWidths=[0.23, 0.1925, 0.1925, 0.1925, 0.1925],
        bbox=(0, -0.015, 1, 0.29),
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor("#F1F5F9" if row == 0 else "white")
    ax.text(
        0,
        -0.11,
        "g1/g2: sparse global/local; g3/g4: dense global/local",
        fontsize=8,
        color=MUTED,
        transform=ax.transAxes,
    )

    heatmap(
        fig,
        (0.385, 0.12, 0.245, 0.275),
        detail["point_heatmaps"]["divergence"],
        "Pointwise divergence\nLighter = sparser",
        "Reds_r",
        minimum=data["divergence_reference_bounds"][0],
        maximum=data["divergence_reference_bounds"][1],
        extend="both",
    )
    heatmap(
        fig,
        (0.71, 0.12, 0.245, 0.275),
        detail["point_heatmaps"]["intrusion"],
        "Territory intrusion\nBoundary and penetration response",
        "Oranges",
    )
    board_legend(fig, 0.48)
    fig.text(
        0.055,
        0.038,
        "Pointwise maps cover all 256 squares. Gridness and intrusion use a 0–1 scale; divergence uses the terminal reference range.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "hexadeca-analysis-iteration-004520.png")


def elo(data):
    fig = plt.figure(figsize=(16, 7), facecolor="white")
    heading(
        fig,
        "Training progress on 16 × 16",
        "Historical checkpoint evaluation  |  Bradley–Terry relative ratings  |  800 simulations per move",
    )
    ax = fig.add_axes((0.085, 0.27, 0.875, 0.535))
    ids = [game["record"]["metadata"]["checkpoint_id"] for game in data["games"]]
    iterations = [int(checkpoint.split("-")[1]) for checkpoint in ids]
    ratings = [data["ratings"][checkpoint] for checkpoint in ids]
    ax.fill_between(
        iterations,
        [r["lower_95"] for r in ratings],
        [r["upper_95"] for r in ratings],
        color="#DBEAFE",
        label="95% bootstrap interval",
    )
    ax.plot(
        iterations,
        [r["rating"] for r in ratings],
        color=BLUE,
        linewidth=3,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2.5,
        label="Internal relative Elo",
    )
    for x, rating in zip(iterations, ratings, strict=True):
        ax.annotate(
            f"{rating['rating']:.0f}",
            (x, rating["rating"]),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            fontsize=13,
            color="#1E40AF",
            weight="bold",
        )
    ax.set_xticks(iterations)
    ax.set_xlabel("Training iteration", labelpad=10)
    ax.set_ylabel("Internal relative Elo", labelpad=12)
    ax.set_ylim(940, 2110)
    ax.margins(x=0.045)
    chart_style(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.135),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.085,
        0.045,
        "Color-balanced matches  |  Anchor: iteration-001360 = 1000  |  Ratings are comparable only within this evaluation pool.",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "hexadeca-relative-elo.png")


if __name__ == "__main__":
    archived = json.loads((ROOT / "showcase-data.json").read_text(encoding="utf-8"))
    comparison(archived)
    terminal_analysis(archived)
    elo(archived)
