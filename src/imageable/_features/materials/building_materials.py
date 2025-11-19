import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from PIL import Image
from shapely.geometry import LineString, Polygon

from imageable._images.camera.camera_parameters import CameraParameters
from imageable._models.huggingface.segformer_segmentation import SegformerSegmentationWrapper
from imageable._models.materials.label_palette import get_material_labels, get_material_palette
from imageable._models.materials.postprocess import colorize_mask
from imageable._models.materials.rmsnet_wrapper import RMSNetSegmentationWrapper
from imageable._utils.geometry.line_geometry import get_angle_between_segments
from imageable._utils.geometry.polygons import get_convex_hull, get_minimum_area_parallelogram
from imageable._utils.geometry.ray_geometry import get_closest_intersected_line
from imageable._utils.geometry.surface_geometry import AreaConversionFactors
from imageable._utils.masks.mask_operations import segment_horizontally_based_on_pixel_density

MODEL_REPO = "urilp4669/Material_Segmentation_Models"
RMSN_WEIGHTS_FILENAME = "rmsnet_split2 (1).pth"

EXPECTED_INTERSECTION_POINTS = 2
MAX_RAY_LENGTH_METERS = 5000.0

def _bearing_to_cartesian_deg(bearing_deg: float) -> float:
    """
    Convert compass bearing (0° = North, 90° = East) to Cartesian angle.

    Parameters
    ----------
    __________
    bearing_deg
        Bearing in degrees.

    Returns
    -------
    _______
    cartesian_angle
        Cartesian angle in degrees (0° = East, 90° = North).
    """
    return (90.0 - bearing_deg) % 360.0


def _wedge_polygon(
    origin_xy: tuple[float, float], bearing_deg: float, hfov_deg: float, radius_m: float = 500.0, n: int = 64
) -> Polygon:
    """
    Create a wedge-shaped polygon (triangle sector)
    representing the camera's field of view.

    Parameters
    ----------
    origin_xy
        The origin point (x, y).
    bearing_deg
        The orientation of the camera as seen from above, in
        degrees and measured clockwise from north.
    hfov_deg
        The horizontal field of view in degrees.
    radius_m
        The radius (length) of the wedge in meters.
    n
        Number of points to approximate the arc.

    Returns
    -------
    wedge
        A Shapely Polygon representing the wedge.
    """
    theta = np.deg2rad(_bearing_to_cartesian_deg(bearing_deg))
    half = np.deg2rad(hfov_deg / 2.0)
    ang = np.linspace(-half, half, n) + theta
    arc = [(origin_xy[0] + radius_m * np.cos(a), origin_xy[1] + radius_m * np.sin(a)) for a in ang]
    return Polygon([origin_xy, *arc, origin_xy])


@dataclass
class BuildingMaterialProperties:
    """Class that encapsulates properties necessary for building material segmentation."""

    img: np.ndarray
    device: str = "cpu"
    rmsnetweights_path: str | None = None
    tile_size: int = 640
    num_classes: int = 20
    backbone: str = "mit_b2"
    sync_bn: bool = False
    backbone_model_path: str | None = None
    building_height: float | None = None
    camera_parameters: CameraParameters | None = None
    footprint: Polygon | None = None
    verbose: bool = False
    alpha: float = 0.5
    display_width: int = 7
    display_height: int = 7
    units: Literal["%", "px", "m2", "ft2", "mi2"] = "%"
    restrict_calculations_to_mask: bool = False
    segformer_name: str = "nvidia/segformer-b5-finetuned-ade-640-640"
    palette_path: str = str(Path(__file__).resolve().parents[2] / "assets" / "ade20k_palette.json")
    device_building_segmentation: str = "cpu"
    remapping_dict: dict[int, int] = field(
        default_factory=lambda: {2: 1, 26: 1, 3: 2, 12: 11, 7: 11, 4: 11, 10: 11, 14: 11}
    )
    shift_building_segmentation_labels: bool = True
    min_restricted_fov_deg: float = 5
    pixel_density_threshold_for_cutting_building: float = 0.15


# ruff: noqa: PLR0911, PLR0912, PLR0915
def get_building_materials_segmentation(properties: BuildingMaterialProperties) -> dict[int, float] | None:
    """
    Perform building material segmentation based on the provided properties.

    Properties
    ----------
    properties
        An instance of BuildingMaterialProperties which contains
        all necessary parameters for material segmentation.

    Returns
    -------
    materials
        A dictionary with material class percentages or areas
        depending on the specified parameter values.
    """
    # Get the percentages
    wrapper = RMSNetSegmentationWrapper(
        backbone=properties.backbone,
        num_classes=properties.num_classes,
        device=properties.device,
        sync_bn=properties.sync_bn,
        weights_path=Path(properties.rmsnetweights_path) if properties.rmsnetweights_path is not None else None,
        tile_size=properties.tile_size,
        model_path=properties.backbone_model_path,
        verbose=properties.verbose
    )
    logits = wrapper.predict(properties.img)
    out = wrapper.postprocess(logits)

    percentages = dict.fromkeys(range(properties.num_classes), 0)
    mask = out["mask"]
    masked_img = None
    if properties.restrict_calculations_to_mask:
        # Get the segformer segmentation mask
        segformer_wrapper = SegformerSegmentationWrapper(
            model_name=properties.segformer_name,
            device=properties.device_building_segmentation,
            palette_path=properties.palette_path,
        )
        segformer_wrapper.load_model()

        all_image_classes_mask = segformer_wrapper.predict(properties.img)
        if all_image_classes_mask is not None:
            if properties.shift_building_segmentation_labels:
                all_image_classes_mask = all_image_classes_mask.squeeze().astype("uint8") + 1
            # Remap the labels
            all_image_classes_mask = segformer_wrapper._remap_labels(all_image_classes_mask, properties.remapping_dict)
            # The label 1 corresponds to buildings.
            # Let's create a binary mask
            building_mask = (all_image_classes_mask == 1).astype(np.uint8)
            # Let's apply the building mask to the material segmentation mask
            mask = mask * building_mask
            masked_img = properties.img * building_mask[:, :, np.newaxis]

    # If the footprint is provided we can further restrict the mask to the field of view
    # of the building
    if (
        properties.restrict_calculations_to_mask
        and properties.footprint is not None
        and properties.camera_parameters is not None
    ):
        percentages: dict[int, float] = dict.fromkeys(range(properties.num_classes), 0.0)

        # Get the observation point
        camera_properties = properties.camera_parameters
        observation_point = (camera_properties.longitude, camera_properties.latitude)

        footprint_coords = list(properties.footprint.exterior.coords)
        footprint_coords = [(p[0], p[1]) for p in footprint_coords]
        hull = get_convex_hull(footprint_coords, close=True)
        parallelogram = get_minimum_area_parallelogram(hull)
        if parallelogram is None:
            parallelogram = hull
        if len(parallelogram) == 0:
            parallelogram = hull
        # We need to get the side of the parallelogram that the visibility ray intersects
        gdf_parallelogram = gpd.GeoDataFrame(geometry=[Polygon(footprint_coords)], crs="EPSG:4326")

        # We need to try to obtain the edge of the footprint that the visibility ray intersects
        cartesian_heading = (90 - properties.camera_parameters.heading) % 360
        direction = (np.cos(np.radians(cartesian_heading)), np.sin(np.radians(cartesian_heading)))

        intersected_segment = get_closest_intersected_line(
            start_point=observation_point, ray_direction=direction, boundaries=gdf_parallelogram
        )

        if len(intersected_segment) == EXPECTED_INTERSECTION_POINTS:
            line1 = [observation_point, intersected_segment[0]]
            line2 = [observation_point, intersected_segment[1]]

            restricted_fov = get_angle_between_segments(line1, line2)
            if restricted_fov is not None:
                original_fov = camera_properties.fov
                if restricted_fov < original_fov and restricted_fov >= properties.min_restricted_fov_deg:
                    mask_width = properties.img.shape[1]
                    x_bound_left = (mask_width / 2) - (mask_width) * (restricted_fov / original_fov)
                    x_bound_right = (mask_width / 2) + (mask_width) * (restricted_fov / original_fov)
                    x_bound_left = int(x_bound_left)
                    x_bound_right = int(x_bound_right)
                    if properties.verbose:
                        print(f"Original FOV: {original_fov}, Restricted FOV: {restricted_fov}")
                        print(f"Restricting calculations to x bounds: {x_bound_left} to {x_bound_right}")

                    mask[:, 0 : int(x_bound_left)] = 0
                    mask[:, int(x_bound_right) :] = 0

                elif restricted_fov < original_fov:
                    mask_width = properties.img.shape[1]
                    x_bound_left = (mask_width / 2) - (mask_width) * (restricted_fov / original_fov)
                    x_bound_left = int(x_bound_left)
                    _, final_column = segment_horizontally_based_on_pixel_density(
                        mask,
                        start_x=x_bound_left,
                        pixel_density_threshold=properties.pixel_density_threshold_for_cutting_building,
                    )
                    mask[:, 0 : int(x_bound_left)] = 0
                    mask[:, int(final_column) :] = 0
                    if properties.verbose:
                        print(
                            f"Restricting calculations based on pixel density starting from column: "
                            f"{x_bound_left}, ending at column: {final_column}"
                        )

    total_pixels = mask.size
    if properties.restrict_calculations_to_mask:
        unique, counts = np.unique(mask[mask != 0], return_counts=True)
        total_pixels = np.sum(counts)
    else:
        unique, counts = np.unique(mask, return_counts=True)

    for u, c in zip(unique, counts, strict=False):
        percentages[int(u)] = float(c) / float(total_pixels)

    # If footprint and height are provided we will calculate the material areas in
    # square meters.
    if properties.verbose:
        palette = get_material_palette()
        labels = get_material_labels()
        colored = colorize_mask(mask, palette)
        base = Image.fromarray(properties.img).resize((colored.shape[1], colored.shape[0]), Image.Resampling.BILINEAR)
        if properties.restrict_calculations_to_mask and masked_img is not None:
            base = Image.fromarray(masked_img).resize((colored.shape[1], colored.shape[0]), Image.Resampling.BILINEAR)

        base_np = np.asarray(base, dtype=np.uint8).copy()
        if properties.restrict_calculations_to_mask and masked_img is not None:
            base_np[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)
            colored[mask == 0] = np.array([0, 0, 0], dtype=np.uint8)

        alpha = properties.alpha
        overlay = (alpha * colored + (1 - alpha) * base_np).astype(np.uint8)

        # Create flexible layout: 2 rows, 2 cols, shorter bottom row
        fig = plt.figure(figsize=(properties.display_width, properties.display_height))
        gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1], figure=fig)

        # Top row: colorized and overlay
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax1.imshow(colored)
        ax1.axis("off")
        ax2.imshow(overlay)
        ax2.axis("off")

        # Bottom row: palette legend spanning both columns
        ax_cb = fig.add_subplot(gs[1, :])
        for i, (name, color) in enumerate(zip(labels, palette, strict=False)):
            ax_cb.add_patch(Rectangle((i, 0), 1, 1, color=color / 255))
            ax_cb.text(
                i + 0.5,
                -0.3,
                name,
                ha="center",
                va="top",
                fontsize=8,
                rotation=90,
            )

        ax_cb.set_xlim(0, len(labels))
        ax_cb.set_ylim(-1, 1)
        ax_cb.axis("off")

        plt.tight_layout()
        plt.show()

    physical_units = False
    if (
        properties.footprint is not None
        and properties.building_height is not None
        and properties.camera_parameters is not None
    ):
        # Let's first get an enclosing parallelogram to the polygon
        footprint_coords = list(properties.footprint.exterior.coords)
        footprint_coords = [(p[0], p[1]) for p in footprint_coords]
        # Get the convex hull
        hull = get_convex_hull(footprint_coords, close=True)
        # Get the minimum area parallelogram
        parallelogram = get_minimum_area_parallelogram(hull)
        if parallelogram is None:
            parallelogram = hull
        if len(parallelogram) == 0:
            parallelogram = hull
        # We need to get the side of the parallelogram that the visibility ray intersects
        gdf_parallelogram = gpd.GeoDataFrame(geometry=[Polygon(parallelogram)], crs="EPSG:4326")
        # I guess I need to prepare everything
        cartesian_heading = (90 - properties.camera_parameters.heading) % 360
        direction = (np.cos(np.radians(cartesian_heading)), np.sin(np.radians(cartesian_heading)))
        observation_point = (properties.camera_parameters.longitude, properties.camera_parameters.latitude)

        closest_line = get_closest_intersected_line(
            start_point=observation_point,
            ray_direction=direction,
            boundaries=gdf_parallelogram,
            max_ray_length=MAX_RAY_LENGTH_METERS,
        )
        # We need to calculate the length of the side in meters
        if len(closest_line) == EXPECTED_INTERSECTION_POINTS:
            line_geom = LineString(closest_line)
            gdf_line = gpd.GeoDataFrame(geometry=[line_geom], crs="EPSG:4326")
            gdf_line = gdf_line.to_crs(epsg=3857)  # Web Mercator, units in meters
            line_length_meters = gdf_line.geometry.length.values[0]
            # Now we can calculate the area of the building side in square meters
            building_side_area_m2 = line_length_meters * properties.building_height
            # Now we can calculate the area per material
            physical_units = True
            for k in percentages:
                percentages[k] = percentages[k] * building_side_area_m2

    if physical_units and properties.units in ["m2", "ft2", "mi2"]:
        if properties.units == "m2":
            labels = get_material_labels()
            return {labels[k]: v for k, v in percentages.items()}
        if properties.units == "ft2":
            for k in percentages:
                percentages[k] = percentages[k] * AreaConversionFactors.SQM_TO_SQFT.value
            labels = get_material_labels()
            return {labels[k]: v for k, v in percentages.items()}
        if properties.units == "mi2":
            for k in percentages:
                percentages[k] = percentages[k] * AreaConversionFactors.SQM_TO_SQMI.value
            labels = get_material_labels()
            return {labels[k]: v for k, v in percentages.items()}

    elif physical_units:
        logging.warning("Physical units not identified. Defaulting to square meters.")
        labels = get_material_labels()
        return {labels[k]: v for k, v in percentages.items()}

    elif properties.units == "%":
        labels = get_material_labels()
        return {labels[k]: v for k, v in percentages.items()}
    elif properties.units == "px":
        for k in list(percentages.keys()):
            percentages[k] = int(percentages[k] * total_pixels)
        labels = get_material_labels()
        return {labels[k]: v for k, v in percentages.items()}
    labels = get_material_labels()
    return {labels[k]: v for k, v in percentages.items()}
