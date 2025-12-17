from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgb

if TYPE_CHECKING:
    from collections.abc import Sequence


class TransparentMaskVisualizer:
    """Overlay segmentation masks on an image with adjustable transparency."""

    @staticmethod
    def _transparent_mask(mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
        """Return an RGBA image where mask==1 pixels get `color` and alpha."""
        rgba = np.zeros((*mask.shape, 4), dtype=float)
        rgba[mask == 1, :3] = color  # RGB (already 0-1 range)
        rgba[mask == 1, 3] = alpha  # A
        return rgba

    @staticmethod
    def _centroid(mask: np.ndarray) -> tuple[float, float]:
        """Centroid (row, col) of a binary mask."""
        ys, xs = np.nonzero(mask)
        return float(ys.mean()), float(xs.mean())

    def visualize(
        self,
        image: np.ndarray,
        masks: Sequence[np.ndarray],
        labels: Sequence[str],
        *,
        figsize: tuple[int, int] = (7, 7),
        title: str = "",
        colors: Sequence[str] | None = None,
        alpha: float = 0.6,
        font_color: str | Sequence[str] = "#ffffff",
        font_size: int = 12,
        save_path: str | None = None,
        dpi: int = 600,
        ax: Any | None = None,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        """
        Plot `image` with semi-transparent `masks` and their `labels`.

        `font_color` can be a single color (applies to all labels) or a list/tuple
        with one color per label.
        """
        if len(masks) != len(labels):
            msg = "`masks` and `labels` must have the same length."
            raise ValueError(msg)

        # ─── handle mask colors ────────────────────────────────────────────────
        if colors is None:
            palette = sns.color_palette("tab20b", 20) + sns.color_palette("tab20c", 20) + sns.color_palette("tab20", 20)
            idx = np.linspace(0, len(palette) - 1, len(masks)).astype(int)
            colors_rgb = [palette[i] for i in idx]
        else:
            colors_rgb = [to_rgb(c) for c in colors]

        # ─── handle font colors ────────────────────────────────────────────────
        if isinstance(font_color, list | tuple | np.ndarray):
            if len(font_color) != len(labels):
                msg = "Length of `font_color` list must match `labels`."
                raise ValueError(msg)
            font_colors = font_color
        else:
            font_colors = [font_color] * len(labels)

        # ─── draw ──────────────────────────────────────────────────────────────
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(image / 255.0 if image.dtype != float or image.max() > 1 else image)

        for mask, label, rgb, fcol in zip(masks, labels, colors_rgb, font_colors, strict=False):
            overlay = self._transparent_mask(mask.astype(bool), rgb, alpha)
            ax.imshow(overlay)
            cy, cx = self._centroid(mask)
            ax.text(
                cx + offset_x,
                cy + offset_y,
                label,
                ha="center",
                va="center",
                fontsize=font_size,
                color=fcol,
            )

        if title:
            ax.set_title(title)
        ax.axis("off")

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight", transparent=True)

        if ax is None:
            plt.show()
