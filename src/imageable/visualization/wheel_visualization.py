from typing import List
import numpy as np
from shapely import Polygon
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Arc
from matplotlib.patches import Wedge
import matplotlib.patheffects as pe


class WheelElement:

    def __init__(self,
                 data: float | np.ndarray | str | Polygon,
                 group_name: str,
                 feature_name: str
                 ):

        self.data = data
        self.group_name = group_name
        self.feature_name = feature_name

    def assign_matrix_indices(self, group_index: int,
                              feature_index: int) -> None:

        self.group_index = group_index
        self.feature_index = feature_index

    def assign_position(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def assign_angular_position(self, radius: float, angle: float) -> None:
        self.radius = radius
        self.angle = angle

    def assign_size(self, size: float) -> None:
        self.size = size

    def draw(self,
             ax: plt.Axes,
             total_radius: float,
             delta_angle: float,
             delta_r: float,
             color: str = "#000000",
             label: str = None,
             gamma: float = 0.8,
             image_scale: float = 0.33) -> None:
        # numeric -> circular segment (thick arc)
        if isinstance(self.data, (float, int)):
            inner_r = self.radius - delta_r / 2
            outer_r = inner_r + delta_r * gamma

            theta1 = (self.angle - delta_angle / 2) * 180 / np.pi
            theta2 = (self.angle + delta_angle / 2) * 180 / np.pi

            wedge = Wedge(
                center=(0, 0),
                r=outer_r,
                theta1=theta1,
                theta2=theta2,
                width=(outer_r - inner_r),
                facecolor=color,
                edgecolor="none",
                alpha=0.95,
            )
            ax.add_patch(wedge)

            if label is not None:
                text_r = outer_r + (0.10 * delta_r)
                tx = text_r * np.cos(self.angle)
                ty = text_r * np.sin(self.angle)
                angle_deg = (np.degrees(self.angle) + 90) % 360
                if 90 < angle_deg < 270:
                    angle_deg += 180
                t = ax.text(
                    tx, ty, label,
                    ha="center", va="center",
                    rotation=angle_deg,
                    rotation_mode="anchor",
                    fontsize=9,
                    color="#111111",
                )
                # subtle outline so it stays readable on top of colors
                t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)])

        # image -> show at element position, square with side = 2*size
        elif isinstance(self.data, np.ndarray):
            size = self.size * image_scale
            img_artist = ax.imshow(
                np.asarray(self.data),
                extent=[
                    self.x - size, self.x + size,
                    self.y - size, self.y + size
                ],
                aspect="equal",
                interpolation="lanczos",
                resample=True,
            )
            # Keep image large but visually discrete by clipping it to its own ring sector.
            clip_pad = 0.98
            inner_r = self.radius - (delta_r / 2) * clip_pad
            outer_r = self.radius + (delta_r / 2) * clip_pad
            theta1 = (self.angle - (delta_angle / 2) * clip_pad) * 180 / np.pi
            theta2 = (self.angle + (delta_angle / 2) * clip_pad) * 180 / np.pi
            clip_wedge = Wedge(
                center=(0, 0),
                r=outer_r,
                theta1=theta1,
                theta2=theta2,
                width=(outer_r - inner_r),
                facecolor="none",
                edgecolor="none",
            )
            ax.add_patch(clip_wedge)
            img_artist.set_clip_path(clip_wedge)

        # text -> write at angular position, oriented
        elif isinstance(self.data, str):
            text_radius = self.radius
            text_x = text_radius * np.cos(self.angle)
            text_y = text_radius * np.sin(self.angle)
            angle_deg = (np.degrees(self.angle) + 90) % 360
            if 90 < angle_deg < 270:
                angle_deg += 180
            ax.text(
                text_x,
                text_y,
                self.data,
                ha="center",
                va="center",
                rotation=angle_deg,
                rotation_mode="anchor"
            )

        # polygon -> just fill it
        elif isinstance(self.data, Polygon):
            xs, ys = self.data.exterior.xy
            centroid_x = 0
            centroid_y = 0
            for x, y in zip(xs, ys):
                centroid_x += x / len(xs)
                centroid_y += y / len(ys)

            # vectors from each point to centroid, scaled to fit
            scaled_xs = [x - centroid_x for x in xs]
            scaled_ys = [y - centroid_y for y in ys]
            norms = [np.sqrt(sx ** 2 + sy ** 2) for sx, sy in zip(scaled_xs, scaled_ys)]
            max_norm = np.max(norms)
            scaled_xs = [sx / max_norm * self.size / 2 for sx in scaled_xs]
            scaled_ys = [sy / max_norm * self.size / 2 for sy in scaled_ys]

            new_xs = [self.x + sx for sx in scaled_xs]
            new_ys = [self.y + sy for sy in scaled_ys]
            ax.fill(new_xs, new_ys, color=color, alpha=0.7)


class WheelVisualization:
    def __init__(self,
                 elements: List[WheelElement],
                 radius_size: float = 10.0,
                 reduction_factor: float = 0.8,
                 image_scale: float = 0.33,
                 image_outer_boost: float = 0.0,
                 ) -> None:

        self.elements = elements

        # groups correspond to angular divisions, features to radial divisions
        def unique_preserve_order(seq):
            seen = set()
            result = []
            for x in seq:
                if x not in seen:
                    seen.add(x)
                    result.append(x)
            return result

        self.group_names = unique_preserve_order([e.group_name for e in elements])
        self.feature_names = unique_preserve_order([e.feature_name for e in elements])
        self.n_groups = len(self.group_names)
        self.n_features = len(self.feature_names)
        self.radius_size = radius_size
        self.reduction_factor = reduction_factor
        if image_scale <= 0:
            raise ValueError("image_scale must be > 0.")
        if image_outer_boost < 0:
            raise ValueError("image_outer_boost must be >= 0.")
        self.image_scale = image_scale
        self.image_outer_boost = image_outer_boost

        self._assign_matrix_indices()
        self._assign_spatial_coordinates()
        self._assign_sizes(reduction_factor=reduction_factor)

    def _assign_matrix_indices(self) -> None:
        for e in self.elements:
            group_index = self.group_names.index(e.group_name)
            feature_index = self.feature_names.index(e.feature_name)
            e.assign_matrix_indices(group_index, feature_index)

    def _assign_spatial_coordinates(self) -> None:
        delta_theta = 2 * np.pi / self.n_groups
        delta_r = self.radius_size / self.n_features
        self.initial_and_final_radii = []
        for e in self.elements:
            theta = delta_theta * (e.group_index + 1 + 1 / 2)
            radius = delta_r * (e.feature_index + 1 + 1 / 2)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            e.assign_position(x, y)
            e.assign_angular_position(radius, theta)
            initial_radius = delta_r * e.feature_index
            final_radius = delta_r * (e.feature_index + 1)
            self.initial_and_final_radii.append((initial_radius, final_radius))

    def _assign_sizes(self, reduction_factor: float = 0.8) -> None:
        index = 0
        delta_angle = 2 * np.pi / self.n_groups
        for e in self.elements:
            initial_radius, final_radius = self.initial_and_final_radii[index]
            index += 1
            max_area = (1 / 2) * (final_radius ** 2 - initial_radius ** 2) * delta_angle
            size = np.sqrt(max_area / np.pi) * reduction_factor
            e.assign_size(size)

    def _get_image_scale_for_element(self, e: WheelElement) -> float:
        if self.n_features <= 1:
            layer_fraction = 1.0
        else:
            layer_fraction = e.feature_index / (self.n_features - 1)
        return self.image_scale * (1.0 + self.image_outer_boost * layer_fraction)

    def draw_wheel(self,
                   cmap: plt.Colormap = plt.cm.viridis,
                   contour_color: str | list[str] = "red",
                   figure_width: float = 8.0,
                   figure_height: float = 8.0,
                   circle_linewidth: float = 0.6,
                   label_features: bool = True,
                   scaffold_alpha: float = 0.1,
                   radial_alpha: float = 0.1,
                   scaffold_color: str = "#111111",
                   footprint_color: str = "#31cceb",
                   show_feature_colorbars: bool = True,
                   save_path: str | None = None,
                   group_label_boost: float = 0.35) -> None:
        center = (0, 0)
        radius_delta = self.radius_size / self.n_features
        radius_list = [i * radius_delta for i in range(self.n_features + 2)]
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))

        if isinstance(contour_color, (list, tuple)):
            if len(contour_color) < len(radius_list):
                raise ValueError("Not enough colors provided for each contour circle.")
            circle_colors = contour_color
        else:
            circle_colors = [contour_color] * len(radius_list)

        for r, c in zip(radius_list, circle_colors):
            circle = plt.Circle(
                center, r,
                color=scaffold_color,
                fill=False,
                lw=circle_linewidth,
                alpha=scaffold_alpha
            )
            ax.add_patch(circle)

        delta_theta = 2 * np.pi / self.n_groups
        angle_list = [i * delta_theta for i in range(self.n_groups)]

        if isinstance(contour_color, (list, tuple)):
            line_color = contour_color[0]
        else:
            line_color = contour_color

        for angle in angle_list:
            x_end = (self.radius_size + radius_delta) * np.cos(angle)
            y_end = (self.radius_size + radius_delta) * np.sin(angle)
            ax.plot(
                [0, x_end], [0, y_end],
                color=scaffold_color,
                lw=circle_linewidth,
                alpha=radial_alpha,
                linestyle="-"
            )

        min_feature_values = np.zeros(len(self.feature_names))
        max_feature_values = np.zeros(len(self.feature_names))
        counted_feature_instances = np.zeros(len(self.feature_names), dtype=int)
        for i, feature_name in enumerate(self.feature_names):
            feature_values = [
                e.data for e in self.elements
                if e.feature_name == feature_name and isinstance(e.data, (float, int))
            ]
            if len(feature_values) > 0:
                min_feature_values[i] = np.min(feature_values)
                max_feature_values[i] = np.max(feature_values)
            else:
                min_feature_values[i] = 0
                max_feature_values[i] = 1

        for e in self.elements:
            if isinstance(e.data, (float, int)):
                label = None
                if label_features:
                    feature_index = self.feature_names.index(e.feature_name)
                    if counted_feature_instances[feature_index] == 0:
                        label = e.feature_name
                        counted_feature_instances[feature_index] += 1

                feature_index = self.feature_names.index(e.feature_name)
                if max_feature_values[feature_index] - min_feature_values[feature_index] > 0:
                    normalized_value = (
                        (e.data - min_feature_values[feature_index]) /
                        (max_feature_values[feature_index] - min_feature_values[feature_index])
                    )
                else:
                    normalized_value = 0.5
                color = cmap(normalized_value)

                e.draw(
                    ax,
                    total_radius=self.radius_size,
                    delta_angle=delta_theta,
                    delta_r=radius_delta,
                    color=color,
                    label=label
                )
            elif isinstance(e.data, np.ndarray):
                e.draw(
                    ax,
                    total_radius=self.radius_size,
                    delta_angle=delta_theta,
                    delta_r=radius_delta,
                    color="black",
                    image_scale=self._get_image_scale_for_element(e)
                )
            elif isinstance(e.data, str):
                e.draw(
                    ax,
                    total_radius=self.radius_size,
                    delta_angle=delta_theta,
                    delta_r=radius_delta,
                    color="black"
                )
            elif isinstance(e.data, Polygon):
                e.draw(
                    ax,
                    total_radius=self.radius_size,
                    delta_angle=delta_theta,
                    delta_r=radius_delta,
                    color=footprint_color
                )

        # group labels
        outer_label_r = self.radius_size + radius_delta * group_label_boost
        tangent_offset = radius_delta * 0.45

        for i, gname in enumerate(self.group_names):
            theta = (2 * np.pi / self.n_groups) * (i + 1 + 1 / 2)

            x = outer_label_r * np.cos(theta)
            y = outer_label_r * np.sin(theta)

            tx = -np.sin(theta)
            ty = np.cos(theta)

            side = 1.0 if np.cos(theta) >= 0 else -1.0
            x += side * tangent_offset * tx
            y += side * tangent_offset * ty

            t = ax.text(
                x, y, gname,
                ha="center", va="center",
                rotation=0,
                fontsize=10,
                fontweight="semibold",
                color="#111111",
                zorder=60,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85
                )
            )
            t.set_path_effects([pe.withStroke(linewidth=2.5, foreground="white", alpha=0.9)])

        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_xlim(-self.radius_size - radius_delta, self.radius_size + radius_delta)
        ax.set_ylim(-self.radius_size - radius_delta, self.radius_size + radius_delta)

        # Per-feature colorbars for numeric rings
        if show_feature_colorbars:
            numeric_features = []
            for i, feature_name in enumerate(self.feature_names):
                values = [
                    e.data for e in self.elements
                    if e.feature_name == feature_name and isinstance(e.data, (float, int))
                ]
                if len(values) > 0:
                    numeric_features.append((feature_name, float(min_feature_values[i]), float(max_feature_values[i])))

            if len(numeric_features) > 0:
                # Bottom layout: compact horizontal bars, wrapped by rows.
                n_cb = len(numeric_features)
                max_per_row = 3
                n_rows = int(np.ceil(n_cb / max_per_row))
                row_block = 0.075
                bottom_pad = 0.04
                cb_height = 0.022
                h_gap = 0.05
                left = 0.07
                right = 0.93
                fig.subplots_adjust(bottom=max(0.14, bottom_pad + n_rows * row_block))

                for j, (fname, vmin, vmax) in enumerate(numeric_features):
                    row = j // max_per_row
                    col = j % max_per_row
                    cols_this_row = min(max_per_row, n_cb - row * max_per_row)
                    avail_w = right - left - h_gap * (cols_this_row - 1)
                    cb_width = avail_w / cols_this_row
                    x = left + col * (cb_width + h_gap)
                    y = bottom_pad + (n_rows - 1 - row) * row_block

                    cax = fig.add_axes([x, y, cb_width, cb_height])
                    if np.isclose(vmin, vmax):
                        vmax = vmin + 1e-9
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    sm = ScalarMappable(norm=norm, cmap=cmap)
                    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
                    cb.set_ticks([vmin, vmax])
                    cb.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
                    cb.ax.tick_params(labelsize=7, pad=1)
                    tick_labels = cb.ax.get_xticklabels()
                    if len(tick_labels) >= 2:
                        tick_labels[0].set_horizontalalignment("left")
                        tick_labels[-1].set_horizontalalignment("right")
                    cb.outline.set_linewidth(0.4)
                    cb.ax.set_title(fname, fontsize=8, pad=1)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
