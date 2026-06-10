from typing import Literal, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch


def display_footprints_categorical(
    gdf: gpd.GeoDataFrame,
    column: str,
    colors: list[str] | None = None,
    category_order: list[str] | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    title_y: float = 1.14,
    subtitle_y: float = 1.03,
    legend_orientation: Literal["vertical", "horizontal"] = "horizontal",
    legend_outside: bool = True,
    legend_y_offset: float = 0.02,
    legend_on_figure: bool = True,
    n_rows_legend: int | None = None,
    title_fontsize: int = 36,
    annot_fontsize: int = 24,
    legend_fontsize: int | None = None,
    annotate_counts: bool = True,
    counts_in_subtitle: bool = False,
    width: int = 12,
    height: int = 10,
    base_gdf: gpd.GeoDataFrame | None = None,
    base_color: str = "#6272a4",
    base_alpha: float = 0.6,
    buffer_meters: float = 0.0,
    base_buffer_meters: float | None = None,
    focus_on_gdf_bounds: bool = True,
    clip_base_to_gdf_bounds: bool = True,
    bounds_pad: float = 0.0,
    add_basemap: bool = False,
    basemap_source: str = "CartoDB.Positron",
    basemap_attribution: bool = False,
    save_path: str | None = None,
    dpi: int = 300,
)-> None:
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
        created_ax = True

    if legend_fontsize is None:
        legend_fontsize = annot_fontsize

    plot_gdf = gdf.copy()
    base_to_plot = base_gdf.copy() if base_gdf is not None else None
    effective_base_buffer = buffer_meters if base_buffer_meters is None else base_buffer_meters

    needs_mercator = (
        buffer_meters > 0
        or (base_to_plot is not None and effective_base_buffer > 0)
        or add_basemap
    )
    if needs_mercator:
        if plot_gdf.crs is None:
            raise ValueError("`gdf.crs` is required when buffering in meters.")
        plot_gdf = plot_gdf.to_crs(epsg=3857)
        if base_to_plot is not None:
            if base_to_plot.crs is None:
                raise ValueError("`base_gdf.crs` is required when buffering in meters.")
            base_to_plot = base_to_plot.to_crs(epsg=3857)

    if buffer_meters > 0:
        plot_gdf["geometry"] = plot_gdf.geometry.buffer(buffer_meters)

    if base_to_plot is not None and effective_base_buffer > 0:
        base_to_plot["geometry"] = base_to_plot.geometry.buffer(effective_base_buffer)

    plot_bounds = None
    if len(plot_gdf) > 0:
        xmin, ymin, xmax, ymax = plot_gdf.total_bounds
        plot_bounds = (
            xmin - bounds_pad,
            ymin - bounds_pad,
            xmax + bounds_pad,
            ymax + bounds_pad,
        )

    if (
        base_to_plot is not None
        and clip_base_to_gdf_bounds
        and plot_bounds is not None
    ):
        xmin, ymin, xmax, ymax = plot_bounds
        base_to_plot = base_to_plot.cx[xmin:xmax, ymin:ymax]

    if focus_on_gdf_bounds and plot_bounds is not None:
        xmin, ymin, xmax, ymax = plot_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if add_basemap:
        try:
            import contextily as ctx
            if basemap_source == "CartoDB.Positron":
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution=False)
            else:
                provider = ctx.providers
                for part in basemap_source.split("."):
                    provider = provider[part]
                ctx.add_basemap(ax, source=provider, attribution=basemap_attribution)
        except Exception as e:
            print(f"basemap failed: {e}")

    if base_to_plot is not None:
        base_to_plot.plot(ax=ax, color=base_color, edgecolor="none", alpha=base_alpha)

    categories = plot_gdf[column].dropna().unique().tolist()
    if category_order is not None:
        ordered_categories = [category for category in category_order if category in categories]
        unordered_categories = [category for category in categories if category not in ordered_categories]
        categories = ordered_categories + unordered_categories
    category_to_color = {}
    if colors is None:
        palette = sns.color_palette("hsv", len(categories))
    else:
        if category_order is not None:
            for i, category in enumerate(category_order):
                if i < len(colors):
                    category_to_color[category] = colors[i]

        if len(colors) >= len(categories):
            palette = colors[:len(categories)]
        else:
            extra_count = len(categories) - len(colors)
            extra_colors = sns.color_palette("hsv", extra_count)
            palette = list(colors) + list(extra_colors)

    if len(category_to_color) > 0:
        for i, category in enumerate(categories):
            if category in category_to_color:
                palette[i] = category_to_color[category]

    legend_handles = []
    subtitle_bits = []
    for i, category in enumerate(categories):
        subgdf = plot_gdf[plot_gdf[column] == category]
        n = len(subgdf)
        subgdf.plot(ax=ax, color=palette[i], edgecolor="none")

        if annotate_counts and not counts_in_subtitle:
            label = f"{category} (n={n:,})"
        else:
            label = str(category)
        legend_handles.append(Patch(facecolor=palette[i], edgecolor="none", label=label))
        subtitle_bits.append(f"{category}: {n:,}")

    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=title_fontsize, y=title_y)

    if subtitle is not None:
        ax.text(
            0.5,
            subtitle_y,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=annot_fontsize,
            family="monospace",
            color="#000000",
        )
    elif annotate_counts and counts_in_subtitle and len(subtitle_bits) > 0:
        ax.text(
            0.5,
            subtitle_y,
            "    ".join(subtitle_bits),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=annot_fontsize,
            family="monospace",
            color="#000000",
        )

    if n_rows_legend is None or n_rows_legend <= 0:
        ncol = len(categories) if legend_orientation == "horizontal" else 1
        legend_rows = 1 if legend_orientation == "horizontal" else len(categories)
    else:
        ncol = max(1, int(np.ceil(len(categories) / n_rows_legend)))
        legend_rows = min(n_rows_legend, len(categories))

    if legend_outside and legend_on_figure:
        fig = cast(Figure, ax.figure)
        y = legend_y_offset if legend_y_offset >= 0 else 0.02
        for existing_legend in list(fig.legends):
            existing_legend.remove()
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            bbox_transform=fig.transFigure,
            ncol=ncol,
            fontsize=legend_fontsize,
            frameon=False,
            handlelength=2.0,
            handleheight=1.2,
        )
        min_bottom = max(0.18, y + 0.08 + max(0, legend_rows - 1) * 0.04)
        if fig.subplotpars.bottom < min_bottom:
            fig.subplots_adjust(bottom=min_bottom)
    else:
        legend_loc = "lower center" if legend_outside else "best"
        legend_anchor = (0.5, legend_y_offset) if legend_outside else None
        ax.legend(
            handles=legend_handles,
            loc=legend_loc,
            bbox_to_anchor=legend_anchor,
            ncol=ncol,
            fontsize=legend_fontsize,
            frameon=False,
            handlelength=2.0,
            handleheight=1.2,
        )

        if created_ax and legend_outside:
            ax.figure.subplots_adjust(bottom=max(0.18, 0.18 + max(0, legend_rows - 1) * 0.04))

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)


def display_footprints_continuous(
    gdf: gpd.GeoDataFrame,
    column: str,
    cmap: str,
    ax: Axes | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    title_y: float = 1.14,
    subtitle_y: float = 1.03,
    title_fontsize: int = 36,
    annot_fontsize: int = 24,
    colorbar_label: str | None = None,
    colorbar_fontsize: int = 16,
    colorbar_orientation: Literal["vertical", "horizontal"] = "vertical",
    min_quantile: float | None = None,
    max_quantile: float | None = None,
    width: int = 12,
    height: int = 10,
    base_gdf: gpd.GeoDataFrame | None = None,
    base_color: str = "#6272a4",
    base_alpha: float = 0.6,
    buffer_meters: float = 0.0,
    base_buffer_meters: float | None = None,
    focus_on_gdf_bounds: bool = True,
    clip_base_to_gdf_bounds: bool = True,
    bounds_pad: float = 0.0,
    add_basemap: bool = False,
    basemap_source: str = "CartoDB.Positron",
    basemap_attribution: bool = False,
    save_path: str | None = None,
    dpi: int = 300,
)-> None:
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
        created_ax = True

    plot_gdf = gdf.copy()
    base_to_plot = base_gdf.copy() if base_gdf is not None else None
    effective_base_buffer = buffer_meters if base_buffer_meters is None else base_buffer_meters

    needs_mercator = (
        buffer_meters > 0
        or (base_to_plot is not None and effective_base_buffer > 0)
        or add_basemap
    )
    if needs_mercator:
        if plot_gdf.crs is None:
            raise ValueError("`gdf.crs` is required when buffering in meters.")
        plot_gdf = plot_gdf.to_crs(epsg=3857)
        if base_to_plot is not None:
            if base_to_plot.crs is None:
                raise ValueError("`base_gdf.crs` is required when buffering in meters.")
            base_to_plot = base_to_plot.to_crs(epsg=3857)

    if buffer_meters > 0:
        plot_gdf["geometry"] = plot_gdf.geometry.buffer(buffer_meters)

    if base_to_plot is not None and effective_base_buffer > 0:
        base_to_plot["geometry"] = base_to_plot.geometry.buffer(effective_base_buffer)

    plot_bounds = None
    if len(plot_gdf) > 0:
        xmin, ymin, xmax, ymax = plot_gdf.total_bounds
        plot_bounds = (
            xmin - bounds_pad,
            ymin - bounds_pad,
            xmax + bounds_pad,
            ymax + bounds_pad,
        )

    if (
        base_to_plot is not None
        and clip_base_to_gdf_bounds
        and plot_bounds is not None
    ):
        xmin, ymin, xmax, ymax = plot_bounds
        base_to_plot = base_to_plot.cx[xmin:xmax, ymin:ymax]

    if focus_on_gdf_bounds and plot_bounds is not None:
        xmin, ymin, xmax, ymax = plot_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if add_basemap:
        try:
            import contextily as ctx
            if basemap_source == "CartoDB.Positron":
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, attribution=False)
            else:
                provider = ctx.providers
                for part in basemap_source.split("."):
                    provider = provider[part]
                ctx.add_basemap(ax, source=provider, attribution=basemap_attribution)
        except Exception as e:
            print(f"basemap failed: {e}")

    if base_to_plot is not None:
        base_to_plot.plot(ax=ax, color=base_color, edgecolor="none", alpha=base_alpha)

    if column not in plot_gdf.columns:
        raise ValueError(f"Column '{column}' not found in GeoDataFrame.")

    values = np.asarray(plot_gdf[column], dtype=float)
    valid_mask = ~np.isnan(values)
    valid_gdf = plot_gdf.loc[valid_mask].copy()
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        raise ValueError(f"Column '{column}' has no valid numeric values to plot.")

    if min_quantile is not None and not (0.0 <= min_quantile <= 1.0):
        raise ValueError("`min_quantile` must be between 0 and 1.")
    if max_quantile is not None and not (0.0 <= max_quantile <= 1.0):
        raise ValueError("`max_quantile` must be between 0 and 1.")
    if min_quantile is not None and max_quantile is not None and min_quantile > max_quantile:
        raise ValueError("`min_quantile` cannot be greater than `max_quantile`.")

    qmin = 0.0 if min_quantile is None else min_quantile
    qmax = 1.0 if max_quantile is None else max_quantile
    vmin = float(np.quantile(valid_values, qmin))
    vmax = float(np.quantile(valid_values, qmax))
    if vmax <= vmin:
        vmin = float(np.min(valid_values))
        vmax = float(np.max(valid_values))
        if vmax <= vmin:
            vmax = vmin + 1e-12

    norm = Normalize(vmin=vmin, vmax=vmax)

    valid_gdf.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        edgecolor="none",
        legend=False,
        vmin=vmin,
        vmax=vmax,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if colorbar_orientation == "horizontal":
        cbar = ax.figure.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02)
    else:
        cbar = ax.figure.colorbar(sm, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar_label = colorbar_label if colorbar_label is not None else column
    cbar.set_label(cbar_label, fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=max(8, colorbar_fontsize - 2))

    ax.set_axis_off()

    if title:
        ax.set_title(title, fontsize=title_fontsize, y=title_y)

    if subtitle is not None:
        ax.text(
            0.5,
            subtitle_y,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=annot_fontsize,
            family="monospace",
            color="#000000",
        )

    if created_ax and colorbar_orientation == "vertical":
        ax.figure.subplots_adjust(right=0.88)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
