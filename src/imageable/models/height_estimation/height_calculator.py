"""
Complete implementation of HeightCalculator.calculate_heights()
This is the main orchestration function that ties everything together.
"""

from dataclasses import dataclass

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .processors.line_classifier import LineClassifier
from .processors.line_refiner import LineRefiner
from .single_view_metrology import SingleViewMetrology
from .vanishing_point import VanishingPointCalculator

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mcolors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
colors_tables = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]


def c(x):
    return sm.to_rgba(x)


@dataclass
class HeightEstimationInput:
    """Data container for height estimation inputs."""

    image: np.ndarray  # Original image array (H, W, 3)
    segmentation: np.ndarray  # Segmentation mask (H, W)
    lines: np.ndarray  # Line segments (N, 2, 2) - N lines with 2 endpoints of 2 coords
    line_scores: np.ndarray  # Line confidence scores (N,)
    vanishing_points: np.ndarray | None = None  # Vanishing points (3, 2) or None
    ground_truth: np.ndarray | None = None  # Ground truth height map (H, W) or None


@dataclass
class CameraParameters:
    """Camera intrinsic parameters."""

    focal_length: float
    cx: float  # Principal point x
    cy: float  # Principal point y
    image_width: int = 640
    image_height: int = 640

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Get camera intrinsic matrix."""
        return np.array([[self.focal_length, 0, self.cx], [0, self.focal_length, self.cy], [0, 0, 1]])

    @classmethod
    def from_fov(cls, fov_degrees: float, image_width: int = 640, image_height: int = 640) -> "CameraParameters":
        """Create camera parameters from field of view."""
        cx = cy = image_width / 2
        focal_length = np.tan(np.deg2rad(fov_degrees / 2.0)) * cx
        return cls(focal_length=focal_length, cx=cx, cy=cy, image_width=image_width, image_height=image_height)


class HeightCalculator:
    """
    Height calculator that works directly with numpy arrays.
    No file I/O - all data passed as parameters.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize calculator with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.svm = SingleViewMetrology()
        self.vp_calc = VanishingPointCalculator()
        self.line_classifier = LineClassifier()
        self.line_refiner = LineRefiner()

    # ruff: noqa: PLR0913, PLR0911
    def calculate_heights(
        self,
        data: HeightEstimationInput,
        camera: CameraParameters,
        pitch: float | None = None,
        use_pitch_only: bool = False,
        use_detected_vpt_only: bool = False,
        verbose: bool = False,
    ) -> dict | None:
        """
        Calculate building heights from input data.

        Args:
            data: Input data container with image, segmentation, lines
            camera: Camera parameters
            pitch: Pitch angle in degrees (optional)
            use_pitch_only: If True, use only pitch for VP calculation
            use_detected_vpt_only: If True, use only provided VPs
            verbose: Enable visualization

        Returns
        -------
            Dictionary with height results or None if processing fails
        """
        try:
            # Get vanishing points
            vps = self._get_vanishing_points(
                data.vanishing_points, camera, pitch, use_pitch_only, use_detected_vpt_only
            )

            if vps is None:
                print("Error: Could not determine vanishing points")
                return None

            # Store calculated VP and vline for later use
            if use_pitch_only or (not use_detected_vpt_only and pitch is not None):
                vertical_v, vline = self.vp_calc.vp_calculation_with_pitch(
                    camera.image_width, camera.image_height, pitch, camera.focal_length
                )
            else:
                vertical_v = None
                vline = None

            # Visualization of inputs (optional)
            if verbose:
                self._visualize_inputs(
                    data.image,
                    data.lines,
                    data.line_scores,
                    data.segmentation,
                    vps,
                    use_pitch_only,
                    vertical_v,
                    vline,
                    camera.image_width,
                )

            # Process line segments
            verticals = self._filter_lines_in_buildings(data.lines, data.line_scores, data.segmentation, vps)

            if not verticals:
                print("No vertical lines found in buildings")
                return None

            # Extend vertical lines to building boundaries
            verticals = self._extend_vertical_lines(verticals, data.segmentation, np.asarray([vps[2, 1], vps[2, 0]]))

            if not verticals:
                print("No lines could be extended to building boundaries")
                return None

            # Calculate heights for each vertical line
            ht_set = self._calculate_line_heights(
                verticals, vps, camera, use_detected_vpt_only, vertical_v, vline, data.ground_truth
            )

            if not ht_set:
                print("No heights could be calculated")
                return None

            # Cluster lines by height
            grouped_lines = self.line_classifier.cluster_lines_with_centers(ht_set, self.config, using_height=True)

            if grouped_lines is None:
                print("No suitable vertical lines found for clustering")
                return None

            # Visualize results
            if verbose:
                self._visualize_results(data.image, data.segmentation, grouped_lines)

            # Prepare output
            heights = []
            for group in grouped_lines:
                median_height = group[-2]
                mean_height = group[-1]
                heights.append(
                    {
                        "median": float(median_height),
                        "mean": float(mean_height),
                        "lines": group[:-2],  # All line data except statistics
                        "count": len(group) - 2,  # Number of lines in group
                    }
                )

            return {
                "heights": heights,
                "num_buildings": len(grouped_lines),
                "vanishing_points": vps.tolist(),
                "total_lines_processed": len(ht_set),
            }

        except Exception as e:
            print(f"Error in height calculation: {e!s}")
            import traceback

            traceback.print_exc()
            return None

    def _get_vanishing_points(
        self,
        provided_vps: np.ndarray | None,
        camera: CameraParameters,
        pitch: float | None,
        use_pitch_only: bool,
        use_detected_vpt_only: bool,
    ) -> np.ndarray | None:
        """
        Get or calculate vanishing points.

        Returns
        -------
            Vanishing points array (3, 2) or None
        """
        if use_pitch_only:
            if pitch is None:
                msg = "Pitch angle required when use_pitch_only=True"
                raise ValueError(msg)

            # Initialize VPs array
            vps = np.zeros([3, 2])

            # Calculate VP from pitch
            vertical_v, _ = self.vp_calc.vp_calculation_with_pitch(
                camera.image_width, camera.image_height, pitch, camera.focal_length
            )

            # Handle special case
            if vertical_v[2] == 0:
                vertical_v[0] = camera.cx
                vertical_v[1] = -9999999

            vps[2, :] = vertical_v[:2]
            return vps

        if provided_vps is not None:
            vps = provided_vps.copy()

            # Optionally override vertical VP with pitch-calculated one
            if not use_detected_vpt_only and pitch is not None:
                vertical_v, _ = self.vp_calc.vp_calculation_with_pitch(
                    camera.image_width, camera.image_height, pitch, camera.focal_length
                )

                if vertical_v[2] == 0:
                    vertical_v[0] = camera.cx
                    vertical_v[1] = -9999999

                vps[2, :] = vertical_v[:2]

            return vps

        return None

    def _filter_lines_in_buildings(
        self, lines: np.ndarray, line_scores: np.ndarray, segmentation: np.ndarray, vps: np.ndarray
    ) -> list:
        """Filter and classify line segments."""
        vert_lines = []
        t_score = float(self.config["LINE_CLASSIFY"]["LineScore"])

        for (a, b), s in zip(lines, line_scores, strict=False):
            if s < t_score:
                continue

            # Check if line is in building area
            if not self.line_classifier.check_if_line_lies_in_building_area(segmentation, a, b, self.config):
                continue

            # Classify as vertical
            is_vert = self.line_classifier.classify_with_vpts(a, b, vps[2], self.config)
            if is_vert:
                vert_lines.append([a, b])

        # Refine and merge vertical lines
        vert_line_refine = []
        for line in vert_lines:
            refined = self.line_refiner.refine_with_vpt([line[0], line[1]], np.asarray([vps[2, 1], vps[2, 0]]))
            vert_line_refine.append(refined)

        # Merge nearby lines
        return self._merge_lines(vert_line_refine)

    def _extend_vertical_lines(self, vertical_lines: list, segmentation: np.ndarray, vptz: np.ndarray) -> list:
        """Extend vertical lines to building boundaries."""
        extd_lines = []

        for line in vertical_lines:
            # Refine with VP
            line = self.line_refiner.refine_with_vpt(line, vptz)

            # Extend to building boundaries
            extd_a, extd_b = self.line_refiner.extend_lines(line[0], line[1], segmentation, self.config)

            if len(extd_a) > 0 and len(extd_b) > 0:
                extd_lines.append([extd_a, extd_b])

        return extd_lines

    def _calculate_line_heights(
        self,
        verticals: list,
        vps: np.ndarray,
        camera: CameraParameters,
        use_detected_vpt_only: bool,
        vertical_v: np.ndarray | None,
        vline: np.ndarray | None,
        ground_truth: np.ndarray | None,
    ) -> list:
        """Calculate height for each vertical line."""
        invK = np.linalg.inv(camera.intrinsic_matrix)
        ht_set = []
        check_list = []

        for line in verticals:
            a = line[0]  # Top point
            b = line[1]  # Bottom point

            # Skip duplicate lines
            if self._is_duplicate(a, b, check_list):
                continue
            check_list.append([a[0], a[1], b[0], b[1]])

            # Convert to homogeneous coordinates (swap x,y)
            a_d3 = np.asarray([a[1], a[0], 1])
            a_d3 = invK @ a_d3

            b_d3 = np.asarray([b[1], b[0], 1])
            b_d3 = invK @ b_d3

            # Calculate height
            if use_detected_vpt_only:
                # Use three detected VPs
                vps0 = np.asarray([vps[0, 0], vps[0, 1], 1])
                vps1 = np.asarray([vps[1, 0], vps[1, 1], 1])
                vps2 = np.asarray([vps[2, 0], vps[2, 1], 1])

                vps0 = invK @ vps0
                vps1 = invK @ vps1
                vps2 = invK @ vps2

                ht = self.svm.sv_measurement(
                    vps0, vps1, vps2, b_d3, a_d3, zc=float(self.config["STREET_VIEW"]["CameraHeight"])
                )
            elif vertical_v is not None and vline is not None:
                # Use cross-ratio with vanishing line
                ht = self.svm.calculate_with_cross_ratio_vl(
                    vline,
                    vertical_v[:2],
                    np.asarray([a[1], a[0]]),
                    np.asarray([b[1], b[0]]),
                    zc=float(self.config["STREET_VIEW"]["CameraHeight"]),
                )
            else:
                continue

            # Get ground truth if available
            ht_gt_org, ht_gt_expd = 0, 0
            if ground_truth is not None:
                ht_gt_org, ht_gt_expd = self._measure_ground_truth(
                    ground_truth, np.asarray([a[1], a[0]]), np.asarray([b[1], b[0]])
                )

            ht_set.append([ht, a, b, ht_gt_org, ht_gt_expd])

        return ht_set

    def _merge_lines(self, lines: list) -> list:
        """Merge nearby parallel lines."""
        merged = []
        used = [False] * len(lines)

        for i in range(len(lines)):
            if used[i]:
                continue

            line_i = lines[i]
            if np.linalg.norm(line_i[0] - line_i[1]) < 10:
                continue

            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue

                line_j = lines[j]
                mid_i = (line_i[0] + line_i[1]) / 2
                mid_j = (line_j[0] + line_j[1]) / 2

                if np.linalg.norm(mid_i - mid_j) < 5:
                    if line_i[0][0] > line_j[0][0]:
                        line_i[0] = line_j[0]
                    if line_i[1][0] < line_j[1][0]:
                        line_i[1] = line_j[1]
                    used[j] = True

            merged.append(line_i)
            used[i] = True

        return merged

    def _is_duplicate(self, a: np.ndarray, b: np.ndarray, check_list: list) -> bool:
        """Check if line (a,b) is duplicate."""
        return any(a0 == a[0] and a1 == a[1] and b0 == b[0] and b1 == b[1] for a0, a1, b0, b1 in check_list)

    def _measure_ground_truth(self, zgt_img: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        """Measure ground truth height if available."""
        if a[1] > b[1]:
            a, b = b, a

        a = np.array(a + [0.5, 0.5], dtype=int)
        b = np.array(b + [0.5, 0.5], dtype=int)

        rows, cols = zgt_img.shape
        a[0] = min(cols - 1, max(0, a[0]))
        a[1] = min(rows - 1, max(0, a[1]))
        b[0] = min(cols - 1, max(0, b[0]))
        b[1] = min(rows - 1, max(0, b[1]))

        if zgt_img[a[1], a[0]] == 0 or zgt_img[b[1], b[0]] == 0:
            gt_org = 0
        else:
            gt_org = abs(zgt_img[a[1], a[0]] - zgt_img[b[1], b[0]])

        return float(gt_org), float(gt_org)

    def _visualize_inputs(
        self,
        image: np.ndarray,
        lines: np.ndarray,
        scores: np.ndarray,
        segmentation: np.ndarray,
        vps: np.ndarray,
        use_pitch_only: bool,
        vertical_v: np.ndarray | None,
        vline: np.ndarray | None,
        w: int,
    ) -> None:
        """Visualize input data."""
        plt.figure(figsize=(10, 8))
        plt.gca().set_axis_off()
        plt.imshow(image)
        plt.imshow(segmentation, alpha=0.5)

        # Draw lines
        t = 0.94
        for (a, b), s in zip(lines, scores, strict=False):
            if s < t:
                continue
            plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
            plt.scatter(a[1], a[0], **PLTOPTS)
            plt.scatter(b[1], b[0], **PLTOPTS)

        # Show vanishing points
        if use_pitch_only and vertical_v is not None and vline is not None:
            x, y = vertical_v[:2]
            plt.scatter(x, y, c="red", s=100, marker="*")
            plt.plot([0, w], [vline[2], vline[2]], c="b", linewidth=3)
        else:
            for i in range(len(vps)):
                x, y = vps[i]
                plt.scatter(x, y, c="red", s=100, marker="*")

        plt.title("Input: Lines, Segmentation, and Vanishing Points")
        plt.show()

    def _visualize_results(self, image: np.ndarray, segmentation: np.ndarray, grouped_lines: list) -> None:
        """Visualize final height results."""
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.imshow(segmentation, alpha=0.3)

        heights = []
        ax_legends = []

        for i, group in enumerate(grouped_lines):
            median_height = group[-2]
            mean_height = group[-1]
            heights.append([median_height, mean_height])

            for j in range(len(group) - 2):
                ht, a, b, *_ = group[j]
                color = colors_tables[i % len(colors_tables)]
                (ax_line,) = plt.plot([a[1], b[1]], [a[0], b[0]], c=color, linewidth=3)
                plt.scatter(a[1], a[0], **PLTOPTS)
                plt.scatter(b[1], b[0], **PLTOPTS)

            if len(group) > 2:
                ax_legends.append(ax_line)

        if ax_legends:
            plt.legend(
                ax_legends, [f"Building {i + 1}: mean={h[1]:.2f}m, median={h[0]:.2f}m" for i, h in enumerate(heights)]
            )

        plt.title("Height Estimation Results")
        plt.show()


class ImageableHeightEstimator:
    """
    Simplified wrapper for integration with imageable's pipeline.
    Works directly with numpy arrays - no file I/O.
    """

    def __init__(self, config: dict | None = None) -> None:
        """
        Initialize with configuration.

        Args:
            config: Configuration dictionary (uses defaults if None)
        """
        self.config = config or self._get_default_config()
        self.calculator = HeightCalculator(self.config)

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "STREET_VIEW": {"HVFoV": "90.0", "CameraHeight": "2.5"},
            "SEGMENTATION": {"SkyLabel": "3", "BuildingLabel": "2", "GroundLabel": "6,7,14"},
            "LINE_CLASSIFY": {"AngleThres": "5.0", "LineScore": "0.94"},
            "LINE_REFINE": {"Edge_Thres": "5,5"},
            "HEIGHT_MEAS": {"MaxDBSANDist": "50.0"},
        }

    def process(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        lines: np.ndarray,
        line_scores: np.ndarray | None = None,
        vanishing_points: np.ndarray | None = None,
        camera_fov: float = 90.0,
        pitch: float = 25.0,
        use_pitch_only: bool = False,
        verbose: bool = False,
    ) -> dict:
        """
        Process image data to estimate building heights.

        Args:
            image: Input image (H, W, 3)
            segmentation: Segmentation mask (H, W)
            lines: Detected lines (N, 2, 2)
            line_scores: Line confidence scores (N,)
            vanishing_points: Optional VPs (3, 2)
            camera_fov: Field of view in degrees
            pitch: Camera pitch in degrees
            use_pitch_only: Use only pitch for VP calculation
            verbose: Enable visualization

        Returns
        -------
            Dictionary with height estimation results
        """
        # Prepare input data
        if line_scores is None:
            line_scores = np.ones(len(lines))

        data = HeightEstimationInput(
            image=image,
            segmentation=segmentation,
            lines=lines,
            line_scores=line_scores,
            vanishing_points=vanishing_points,
        )

        # Setup camera
        h, w = image.shape[:2]
        camera = CameraParameters.from_fov(camera_fov, w, h)

        # Calculate heights
        result = self.calculator.calculate_heights(
            data=data,
            camera=camera,
            pitch=pitch,
            use_pitch_only=use_pitch_only or (vanishing_points is None),
            use_detected_vpt_only=False,
            verbose=verbose,
        )

        return result or {"error": "Height calculation failed", "heights": []}
