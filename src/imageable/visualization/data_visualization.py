from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

DRACULA_BG = "#282a36"
DRACULA_FG = "#f8f8f2"
DRACULA_GRID = "#44475a"
DRACULA_PALETTE = [
    "#8be9fd",  # cyan
    "#ff79c6",  # pink
    "#bd93f9",  # purple
    "#50fa7b",  # green
    "#ffb86c",  # orange
    "#f1fa8c",  # yellow
]


def _natural_sort_key(value: str) -> tuple[int | str, ...]:
    parts = re.split(r"(\d+)", str(value))
    key: list[int | str] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def _to_matrix(
    values: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray,
    model_names: Sequence[str],
    n_groups: int,
    value_name: str,
) -> np.ndarray:
    if isinstance(values, Mapping):
        matrix = np.asarray([values[m] for m in model_names], dtype=float)
    else:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[np.newaxis, :]

    expected_shape = (len(model_names), n_groups)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"`{value_name}` must have shape {expected_shape}, got {matrix.shape}."
        )
    return matrix


def plot_grouped_distribution_bars(
    groups: Sequence[str],
    model_names: Sequence[str],
    rmse: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray,
    *,
    counts: Sequence[int] | None = None,
    target_mean: Sequence[float] | None = None,
    title: str = "",
    x_label: str = "Group",
    y_label: str = "Value",
    order: Literal["input", "natural", "rmse_mean", "n"] = "natural",
    fixed_groups: Sequence[str] | None = None,
    show_n: bool = True,
    show_target_mean: bool = False,
    legend_outside: bool = False,
    ylim_pad: float = 0.08,
    savepath: str | None = None,
    use_short_xticks: bool = False,
    label_mode: Literal["all", "delta_vs_best", "none"] = "all",
    label_pad_frac: float = 0.04,
    error_mode: Literal["std", "ci"] | None = None,
    rmse_std: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray | None = None,
    rmse_ci_low: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray | None = None,
    rmse_ci_high: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | np.ndarray | None = None,
    colors: Sequence[str] | None = None,
    model_labels: Mapping[str, str] | None = None,
    group_labels: Mapping[str, str] | None = None,
    figsize: tuple[float, float] = (8.8, 4.6),
    show: bool = True,
    dpi:int = 300
)->tuple[Figure, Axes]:
    """
    Grouped distribution bars with simple array/dict inputs.

    Parameters
    ----------
    groups
        Group names, length = n_groups.
    model_names
        Model keys, length = n_models.
    rmse
        Either:
        - dict: {model_name: [value per group]}, or
        - 2D array-like with shape (n_models, n_groups).
    counts
        Optional per-group n values.
    target_mean
        Optional per-group target means.
    error_mode
        - None: no error bars
        - "std": uses `rmse_std` (same shape as rmse)
        - "ci": uses `rmse_ci_low` / `rmse_ci_high` (same shape as rmse)
    """
    if len(groups) == 0:
        raise ValueError("`groups` cannot be empty.")
    if len(model_names) == 0:
        raise ValueError("`model_names` cannot be empty.")

    n_groups = len(groups)
    group_names = [str(g) for g in groups]
    model_names = [str(m) for m in model_names]

    if counts is not None and len(counts) != n_groups:
        raise ValueError("`counts` must have same length as `groups`.")
    if target_mean is not None and len(target_mean) != n_groups:
        raise ValueError("`target_mean` must have same length as `groups`.")

    value_matrix = _to_matrix(rmse, model_names, n_groups, "values")

    if error_mode == "std":
        if rmse_std is None:
            raise ValueError("`error_mode='std'` requires `rmse_std`.")
        std_matrix = _to_matrix(rmse_std, model_names, n_groups, "std")
        ci_low_matrix = None
        ci_high_matrix = None
    elif error_mode == "ci":
        if rmse_ci_low is None or rmse_ci_high is None:
            raise ValueError("`error_mode='ci'` requires `rmse_ci_low` and `rmse_ci_high`.")
        ci_low_matrix = _to_matrix(rmse_ci_low, model_names, n_groups, "ci_low")
        ci_high_matrix = _to_matrix(rmse_ci_high, model_names, n_groups, "ci_high")
        std_matrix = None
    else:
        std_matrix = None
        ci_low_matrix = None
        ci_high_matrix = None

    # Group ordering
    name_to_idx = {name: i for i, name in enumerate(group_names)}
    if fixed_groups is not None:
        order_idx = [name_to_idx[g] for g in fixed_groups if g in name_to_idx]
    elif order == "rmse_mean":
        order_idx = list(np.argsort(np.nanmean(value_matrix, axis=0)))
    elif order == "n" and counts is not None:
        order_idx = list(np.argsort(np.asarray(counts))[::-1])
    elif order == "input":
        order_idx = list(range(n_groups))
    elif order == "natural":
        order_idx = sorted(range(n_groups), key=lambda i: _natural_sort_key(group_names[i]))
    else:
        order_idx = sorted(range(n_groups), key=lambda i: group_names[i])

    group_names = [group_names[i] for i in order_idx]
    value_matrix = value_matrix[:, order_idx]
    if std_matrix is not None:
        std_matrix = std_matrix[:, order_idx]
    if ci_low_matrix is not None and ci_high_matrix is not None:
        ci_low_matrix = ci_low_matrix[:, order_idx]
        ci_high_matrix = ci_high_matrix[:, order_idx]
    if counts is not None:
        counts = [counts[i] for i in order_idx]
    if target_mean is not None:
        target_mean = [target_mean[i] for i in order_idx]

    pretty_models = [model_labels.get(m, m) if model_labels else m for m in model_names]
    pretty_groups = [group_labels.get(g, g) if group_labels else g for g in group_names]

    # Tick labels
    if use_short_xticks:
        tick_labels = [f"G{i+1}" for i in range(len(pretty_groups))]
        short_map = dict(zip(tick_labels, pretty_groups, strict=False))
    else:
        tick_labels = pretty_groups
        short_map = None

    if show_n or show_target_mean:
        with_meta = []
        for i, lbl in enumerate(tick_labels):
            bits = []
            if show_target_mean and target_mean is not None:
                bits.append(f"μ={target_mean[i]:.2f}")
            if show_n and counts is not None:
                bits.append(f"n={int(counts[i])}")
            with_meta.append(f"{lbl}\n({' | '.join(bits)})" if bits else lbl)
        tick_labels = with_meta

    # Colors
    if colors is None:
        color_list = DRACULA_PALETTE
    else:
        color_list = list(colors)
    if len(color_list) < len(model_names):
        repeats = int(np.ceil(len(model_names) / len(color_list)))
        color_list = (color_list * repeats)[: len(model_names)]

    # Plot
    x = np.arange(len(group_names))
    n_models = len(model_names)
    group_width = 0.86
    bar_width = group_width / max(1, n_models)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    bg_color = "white"
    fg_color = "#111827"
    grid_color = "#d1d5db"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    containers = []
    for j, model in enumerate(model_names):
        x_pos = x - (group_width / 2) + (j + 0.5) * bar_width
        y = value_matrix[j]

        yerr = None
        if error_mode == "std" and std_matrix is not None:
            yerr = std_matrix[j]
        elif error_mode == "ci" and ci_low_matrix is not None and ci_high_matrix is not None:
            lower = np.maximum(0.0, y - ci_low_matrix[j])
            upper = np.maximum(0.0, ci_high_matrix[j] - y)
            yerr = np.vstack([lower, upper])

        bars = ax.bar(
            x_pos,
            y,
            width=bar_width,
            color=color_list[j],
            edgecolor="none",
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
            label=pretty_models[j],
        )
        containers.append(bars)

    # Style
    ax.set_title(title, color=fg_color)
    ax.set_xlabel(x_label, color=fg_color)
    ax.set_ylabel(y_label, color=fg_color)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.5, linewidth=0.8, color=grid_color)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(fg_color)
    ax.spines["bottom"].set_color(fg_color)
    ax.tick_params(axis="x", colors=fg_color)
    ax.tick_params(axis="y", colors=fg_color)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=0 if use_short_xticks else 20, ha="center" if use_short_xticks else "right")

    max_value = float(np.nanmax(value_matrix))
    label_pad = max_value * max(0.0, float(label_pad_frac))
    extra_top = label_pad * (0.45 if label_mode == "all" else (1.8 if label_mode != "none" else 0.0))
    if np.isfinite(max_value):
        ax.set_ylim(0, max_value * (1 + max(0.0, float(ylim_pad))) + extra_top)

    # Labels
    if label_mode == "all":
        for c in containers:
            ax.bar_label(c, fmt="%.2f", padding=2, fontsize=8, color=fg_color)
    elif label_mode == "delta_vs_best":
        for i in range(len(group_names)):
            vals = value_matrix[:, i].astype(float)
            best_j = int(np.nanargmin(vals))
            best_val = float(vals[best_j])

            rect = containers[best_j].patches[i]
            x_best = rect.get_x() + rect.get_width() / 2
            y_best = rect.get_height()
            ax.text(x_best, y_best + label_pad, f"{best_val:.2f}", ha="center", va="bottom", fontsize=8, color=fg_color, fontweight="bold")

            for j in range(n_models):
                if j == best_j:
                    continue
                r = containers[j].patches[i]
                xj = r.get_x() + r.get_width() / 2
                yj = r.get_height()
                delta = float(vals[j] - best_val)
                ax.text(xj, yj + label_pad, f"+{delta:.2f}", ha="center", va="bottom", fontsize=7, color=fg_color)

    # Legend
    if legend_outside:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), labelcolor=fg_color)
    else:
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.28),
            ncol=max(1, min(n_models, 4)),
            columnspacing=1.0,
            handlelength=1.6,
            labelcolor=fg_color,
        )

    if short_map is not None:
        mapping_lines = [f"{k}: {short_map[k]}" for k in sorted(short_map.keys(), key=_natural_sort_key)]
        fig.text(0.01, -0.02, "  |  ".join(mapping_lines), ha="left", va="top", fontsize=8, color=fg_color)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", pad_inches=0.03, dpi = 300)

    if show:
        plt.show()
    return fig, ax


def plot_styled_histogram(
    values: Sequence[float] | np.ndarray,
    *,
    title: str,
    x_label: str = "RMSE Difference",
    y_label: str = "Frequency",
    bins: int = 10,
    color: str = "#bd93f9",
    edgecolor: str = DRACULA_GRID,
    reference_x: float | None = 0.0,
    reference_color: str = "#ff5555",
    figsize: tuple[float, float] = (9.5, 6.5),
    title_fontsize: int = 23,
    axis_fontsize: int = 18,
    tick_fontsize: int = 14,
    annotation_fontsize: int = 15,
    better_left_label: str | None = None,
    better_right_label: str | None = None,
    left_annotation_x: float = 0.02,
    right_annotation_x: float = 0.98,
    annotation_y: float = 0.965,
    left_annotation_color: str = DRACULA_GRID,
    right_annotation_color: str = "#ff5555",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    savepath: str | None = None,
    show: bool = True,
    dpi:int = 300,
)->tuple[Figure, Axes]:
    """Plot a histogram with the same high-readability style used in notebooks."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("`values` cannot be empty.")

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.hist(arr, bins=bins, color=color, edgecolor=edgecolor)

    if reference_x is not None:
        ax.axvline(reference_x, color=reference_color, linestyle=":", linewidth=2.6)

    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=16)
    ax.set_xlabel(x_label, fontsize=axis_fontsize, labelpad=8)
    ax.set_ylabel(y_label, fontsize=axis_fontsize, labelpad=8)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.25, color=DRACULA_GRID)

    if better_left_label is not None and reference_x is not None:
        left_pct = float((arr < reference_x).mean() * 100)
        ax.text(
            left_annotation_x,
            annotation_y,
            f"Better: \n{better_left_label} ({left_pct:.1f}%)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=left_annotation_color,
            fontsize=annotation_fontsize,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=left_annotation_color, alpha=0.95),
        )

    if better_right_label is not None and reference_x is not None:
        right_pct = float((arr > reference_x).mean() * 100)
        ax.text(
            right_annotation_x,
            annotation_y,
            f"Better: \n{better_right_label} ({right_pct:.1f}%)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=right_annotation_color,
            fontsize=annotation_fontsize,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=right_annotation_color, alpha=0.95),
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", pad_inches=0.12)

    if show:
        plt.show()
    return fig, ax
