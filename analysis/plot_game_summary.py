"""Render a compact visual summary for one Hexadeca game record.

The figure combines the final board, the two G0 local-grid heatmaps, and every
scalar/vector feature produced by :mod:`analysis.features`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch, Rectangle

from analysis import (
    FeatureResult,
    HexadecaFeatureParser,
    terminal_divergence_reference_bounds,
)


BLUE = "#2563EB"
BLUE_LIGHT = "#BFDBFE"
ORANGE = "#F97316"
ORANGE_LIGHT = "#FED7AA"
NEUTRAL = "#F3F4F6"
GRID = "#CBD5E1"
TEXT = "#111827"
TERMINAL_DIVERGENCE_PLOT_MIN, TERMINAL_DIVERGENCE_PLOT_MAX = (
    terminal_divergence_reference_bounds()
)


def _font() -> FontProperties:
    """Use a Windows CJK font when available, with a portable fallback."""

    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if candidate.is_file():
            return FontProperties(fname=str(candidate))
    return FontProperties(family="DejaVu Sans")


def _draw_board(
    ax: plt.Axes,
    matrix: tuple[tuple[int, ...], ...],
    font: FontProperties,
) -> None:
    size = len(matrix)
    for row in range(size):
        for column in range(size):
            value = matrix[row][column]
            if value == 2:
                color = BLUE_LIGHT
            elif value == -2:
                color = ORANGE_LIGHT
            else:
                color = "#FFFFFF" if value == 0 else "#FFFFFF"
            ax.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor=GRID,
                    linewidth=0.45,
                )
            )
            if value == 1:
                ax.add_patch(plt.Circle((column, row), 0.31, color=BLUE, zorder=3))
            elif value == -1:
                ax.add_patch(plt.Circle((column, row), 0.31, color=ORANGE, zorder=3))

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect("equal")
    step = 2 if size <= 16 else 3
    ax.set_xticks(range(0, size, step))
    ax.set_yticks(range(0, size, step))
    ax.tick_params(labelsize=8, colors=TEXT)
    ax.set_xlabel("列坐标", fontproperties=font, color=TEXT)
    ax.set_ylabel("行坐标", fontproperties=font, color=TEXT)
    ax.set_title("终局棋盘", fontproperties=font, fontsize=15, color=TEXT, pad=10)
    legend = (
        Patch(facecolor=BLUE, edgecolor="none", label="蓝棋子"),
        Patch(facecolor=ORANGE, edgecolor="none", label="橙棋子"),
        Patch(facecolor=BLUE_LIGHT, edgecolor=GRID, label="蓝领地空位"),
        Patch(facecolor=ORANGE_LIGHT, edgecolor=GRID, label="橙领地空位"),
        Patch(facecolor="white", edgecolor=GRID, label="中立空位"),
    )
    legend_font = font.copy()
    legend_font.set_size(8)
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        prop=legend_font,
    )


def _draw_heatmap(
    ax: plt.Axes,
    values: tuple[tuple[int, ...], ...],
    title: str,
    maximum: float,
    font: FontProperties,
) -> None:
    cmap = LinearSegmentedColormap.from_list(
        "gridness",
        ["#EFF6FF", "#93C5FD", "#2563EB", "#172554"],
    )
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=maximum, interpolation="nearest")
    ax.set_title(title, fontproperties=font, fontsize=13, color=TEXT, pad=8)
    ax.set_xlabel("窗口起点列", fontproperties=font, color=TEXT)
    ax.set_ylabel("窗口起点行", fontproperties=font, color=TEXT)
    ax.tick_params(labelsize=8, colors=TEXT)
    step = 2 if len(values) <= 11 else 3
    ax.set_xticks(range(0, len(values[0]), step))
    ax.set_yticks(range(0, len(values), step))
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("局部格度值", fontproperties=font, color=TEXT)
    colorbar.ax.tick_params(labelsize=8, colors=TEXT)


def _draw_point_heatmap(
    ax: plt.Axes,
    values: tuple[tuple[float, ...], ...],
    title: str,
    colors: tuple[str, ...],
    colorbar_label: str,
    font: FontProperties,
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    extend: str = "neither",
) -> None:
    cmap = LinearSegmentedColormap.from_list(title, list(colors))
    image = ax.imshow(
        values,
        cmap=cmap,
        vmin=minimum,
        vmax=maximum,
        interpolation="nearest",
    )
    ax.set_title(title, fontproperties=font, fontsize=13, color=TEXT, pad=8)
    ax.set_xlabel("列坐标", fontproperties=font, color=TEXT)
    ax.set_ylabel("行坐标", fontproperties=font, color=TEXT)
    size = len(values)
    step = max(1, size // 4)
    ax.set_xticks(tuple(range(0, size, step)))
    ax.set_yticks(tuple(range(0, size, step)))
    ax.tick_params(labelsize=8, colors=TEXT)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, extend=extend)
    colorbar.set_label(colorbar_label, fontproperties=font, color=TEXT)
    colorbar.ax.tick_params(labelsize=8, colors=TEXT)


def _draw_metrics(ax: plt.Axes, result: FeatureResult, font: FontProperties) -> None:
    ax.axis("off")
    ax.set_title("五度参数", fontproperties=font, fontsize=15, color=TEXT, pad=10)
    ax.text(
        0.02,
        0.965,
        f"着法 {result.move_count} 步  ·  蓝 {result.piece_counts[1]}  ·  橙 {result.piece_counts[2]}",
        transform=ax.transAxes,
        fontproperties=font,
        fontsize=10,
        color="#4B5563",
        va="top",
    )

    rows: list[tuple[str, str]] = [
        ("D 散度", f"{result.D:.9f}"),
        ("I 侵度", "未定义" if result.I is None else f"{result.I:.9f}"),
        ("S0 主对称度", f"{result.S0:.9f}"),
        ("S1 蓝对称度", f"{result.S1:.9f}"),
        ("S2 橙对称度", f"{result.S2:.9f}"),
        ("B0 总贴边度", f"{result.B0:.9f}"),
        ("B1 蓝贴边度", f"{result.B1:.9f}"),
        ("B2 橙贴边度", f"{result.B2:.9f}"),
    ]
    y = 0.88
    for label, value in rows:
        ax.text(
            0.04,
            y,
            label,
            transform=ax.transAxes,
            fontproperties=font,
            fontsize=10.5,
            color=TEXT,
            va="center",
        )
        ax.text(
            0.96,
            y,
            value,
            transform=ax.transAxes,
            fontproperties=font,
            fontsize=10.5,
            color=TEXT,
            va="center",
            ha="right",
            family="DejaVu Sans Mono",
        )
        y -= 0.052

    ax.text(
        0.04,
        y - 0.01,
        "G 格度（g1 全局 · g2 局部 · g3 全局 · g4 局部）",
        transform=ax.transAxes,
        fontproperties=font,
        fontsize=10.5,
        color=TEXT,
        va="center",
    )
    y -= 0.075
    for name, values in (
        ("G0 总体", result.G0),
        ("G1 蓝棋", result.G1),
        ("G2 橙棋", result.G2),
    ):
        ax.text(
            0.06,
            y,
            name,
            transform=ax.transAxes,
            fontproperties=font,
            fontsize=10.5,
            color=TEXT,
            va="center",
        )
        ax.text(
            0.96,
            y,
            "[" + ", ".join(f"{value:.6f}" for value in values) + "]",
            transform=ax.transAxes,
            fontproperties=font,
            fontsize=9.4,
            color=TEXT,
            va="center",
            ha="right",
            family="DejaVu Sans Mono",
        )
        y -= 0.064

    ax.text(
        0.04,
        0.035,
        "1/-1 为棋子；2/-2 为禁入点。逐点热力图覆盖全部 256 点。",
        transform=ax.transAxes,
        fontproperties=font,
        fontsize=9,
        color="#6B7280",
        va="bottom",
    )


def render_summary(record_path: Path, output_path: Path) -> None:
    import json

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    board_size = len(payload.get("final_matrix", []))
    parser = HexadecaFeatureParser(board_size=board_size)
    record = parser.parse(record_path)
    result = parser.analyze(record)
    divergence_min, divergence_max = terminal_divergence_reference_bounds(board_size)

    font = _font()
    fig = plt.figure(figsize=(24, 11.5), facecolor="white")
    grid = fig.add_gridspec(
        nrows=2,
        ncols=4,
        width_ratios=(1.12, 1.0, 1.0, 1.2),
        height_ratios=(1, 1),
        wspace=0.34,
        hspace=0.34,
    )
    board_ax = fig.add_subplot(grid[:, 0])
    sparse_ax = fig.add_subplot(grid[0, 1])
    dense_ax = fig.add_subplot(grid[1, 1])
    divergence_ax = fig.add_subplot(grid[0, 2])
    intrusion_ax = fig.add_subplot(grid[1, 2])
    metrics_ax = fig.add_subplot(grid[:, 3])

    _draw_board(board_ax, record.final_matrix, font)
    _draw_heatmap(
        sparse_ax,
        result.heatmaps[0].sparse,
        f"G0 疏晶格局部格度（6×6 窗口，{len(result.heatmaps[0].sparse)}×{len(result.heatmaps[0].sparse)}）",
        1.0,
        font,
    )
    _draw_heatmap(
        dense_ax,
        result.heatmaps[0].dense,
        f"G0 密晶格局部格度（4×4 窗口，{len(result.heatmaps[0].dense)}×{len(result.heatmaps[0].dense)}）",
        1.0,
        font,
    )
    _draw_point_heatmap(
        divergence_ax,
        result.point_heatmaps.divergence,
        f"逐点散度热力图（{board_size}×{board_size}）",
        # Divergence is larger in sparser regions, so use a light colour for
        # high values and a dark colour for dense regions.
        ("#7F1D1D", "#EF4444", "#FCA5A5", "#FFF5F5"),
        "逐点散度（终局参照范围）",
        font,
        minimum=divergence_min,
        maximum=divergence_max,
        extend="both",
    )
    _draw_point_heatmap(
        intrusion_ax,
        result.point_heatmaps.intrusion,
        f"终局领地侵度热力图（{board_size}×{board_size}）",
        ("#FFF7ED", "#FDBA74", "#F97316", "#9A3412"),
        "交界与打入响应（0–1）",
        font,
    )
    _draw_metrics(metrics_ax, result, font)

    fig.suptitle(
        "Hexadeca 五度参数 · 终局分析",
        fontproperties=font,
        fontsize=20,
        color=TEXT,
        y=0.985,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("record", type=Path)
    argument_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
    )
    arguments = argument_parser.parse_args()
    output = arguments.output or Path("analysis/png") / (
        arguments.record.stem + "-summary.png"
    )
    render_summary(arguments.record, output)
    print(output)


if __name__ == "__main__":
    main()
