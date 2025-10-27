"""
Main building property extraction orchestrator.

This module provides the main function to extract all properties from a building
given its polygon and optional image/model data.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import Polygon

from .building import BuildingProperties
from .footprint import extract_footprint_properties
from .image import ImageCalculator


def extract_building_properties(
    building_id: str,
    polygon: Polygon,
    all_buildings: list[Polygon] | None = None,
    neighbor_radius: float = 600.0,
    crs: int = 4326,
    # Image-based properties (optional)
    street_view_image: np.ndarray | None = None,
    building_mask: np.ndarray | None = None,
    window_mask: np.ndarray | None = None,
    door_mask: np.ndarray | None = None,
    # Height estimation (optional)
    height_value: float | None = None,
    # Material segmentation (optional)
    material_percentages: dict[str, float] | None = None,
    verbose: bool = False,
) -> BuildingProperties:
    """
    Extract all properties for a building from its polygon and optional data.

    This is the main function you should use. It orchestrates extraction of:
    - Footprint properties (geometric, engineered, contextual)
    - Image properties (color, shape, façade) - if image provided
    - Height - if provided
    - Materials - if provided

    Parameters
    ----------
    building_id
        Unique identifier for the building.
    polygon
        Building footprint as Shapely Polygon.
    all_buildings
        List of all building polygons for neighbor detection. Optional.
    neighbor_radius
        Max distance for neighbor detection (meters). Default 600.
    crs
        Coordinate reference system. Default 4326 (WGS84).
    street_view_image
        RGB image array (H, W, 3). Optional.
    building_mask
        Binary mask of building in image (H, W). Optional.
    window_mask
        Binary mask of windows (H, W). Optional.
    door_mask
        Binary mask of doors (H, W). Optional.
    height_value
        Pre-calculated building height in meters. Optional.
    material_percentages
        Pre-calculated material percentages. Optional.

    Returns
    -------
    BuildingProperties
        Complete property object with all extracted features.

    Examples
    --------
    Minimal usage (just polygon):
    >>> props = extract_building_properties("b001", polygon)
    >>> print(props.projected_area)

    With neighbors:
    >>> props = extract_building_properties("b001", polygon, all_buildings=all_polygons)
    >>> print(props.neighbor_count)

    With everything:
    >>> props = extract_building_properties(
    ...     "b001", polygon,
    ...     all_buildings=all_polygons,
    ...     street_view_image=image,
    ...     building_mask=mask,
    ...     height_value=25.3,
    ...     material_percentages={'concrete': 45.2, 'glass': 30.1}
    ... )
    """
    # Initialize properties container
    properties = BuildingProperties(building_id=building_id)

    # ========== 1. Footprint Properties ==========
    if verbose:
        print(f"[1/4] Extracting footprint properties for {building_id}...")

    footprint_props = extract_footprint_properties(
        polygon=polygon, all_footprints=all_buildings, crs=crs, neighbor_radius=neighbor_radius
    )
    properties.update_footprint_features(footprint_props)
    if verbose:
        print(f"  ✓ Area: {properties.projected_area:.2f} m²")
        print(f"  ✓ Complexity: {properties.complexity:.4f}")
    if all_buildings and verbose:
        print(f"  ✓ Neighbors: {properties.neighbor_count}")

    # ========== 2. Height ==========
    if verbose:
        print("[2/4] Processing height...")

    if height_value is not None:
        properties.update_height(height_value)
        if verbose:
            print(f"  ✓ Height: {height_value:.2f} m")
    elif verbose:
        print("  ⊘ No height provided")

    # ========== 3. Material Percentages ==========
    if verbose:
        print("[3/4] Processing materials...")

    if material_percentages is not None:
        properties.update_material_percentages(material_percentages)
        if verbose:
            print(f"  ✓ Materials: {len(material_percentages)} types")
    elif verbose:
        print("  ⊘ No materials provided")

    # ========== 4. Image Properties ==========
    if verbose:
        print("[4/4] Extracting image properties...")

    if street_view_image is not None and building_mask is not None:
        image_calc = ImageCalculator(img=street_view_image, building_mask=building_mask)

        image_features = image_calc.extract_all_features(window_mask=window_mask, door_mask=door_mask)
        properties.update_image_features(image_features)

        if verbose:
            print("  ✓ Color features extracted")
        if verbose:
            print("  ✓ Shape features extracted")
        if window_mask is not None or door_mask is not None:
            if verbose:
                print("  ✓ Façade features extracted")
    elif verbose:
        print("  ⊘ No image/mask provided")
    if verbose:
        print(f"\n✓ Complete! Total features: {len(properties.get_feature_vector())}")

    return properties


def batch_extract_properties(
    buildings: list[dict], neighbor_radius: float = 600.0, crs: int = 4326, verbose: bool = True
) -> list[BuildingProperties]:
    """
    Extract properties for multiple buildings in batch.

    Parameters
    ----------
    buildings
        List of dictionaries, each containing:
        - 'id': building identifier (required)
        - 'polygon': Shapely Polygon (required)
        - 'image': RGB image array (optional)
        - 'mask': building mask (optional)
        - 'height': height value (optional)
        - 'materials': material percentages dict (optional)
        - 'window_mask': window mask (optional)
        - 'door_mask': door mask (optional)
    neighbor_radius
        Max distance for neighbor detection.
    crs
        Coordinate reference system.
    verbose
        Print progress.

    Returns
    -------
    List[BuildingProperties]
        List of property objects for all buildings.

    Example
    -------
    >>> buildings = [
    ...     {'id': 'b001', 'polygon': poly1, 'height': 25.0},
    ...     {'id': 'b002', 'polygon': poly2, 'height': 18.5},
    ... ]
    >>> all_props = batch_extract_properties(buildings)
    """
    all_properties = []
    all_polygons = [b["polygon"] for b in buildings]

    for i, building in enumerate(buildings):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Building {i + 1}/{len(buildings)}: {building['id']}")
            print(f"{'=' * 60}")

        props = extract_building_properties(
            building_id=building["id"],
            polygon=building["polygon"],
            all_buildings=all_polygons,
            neighbor_radius=neighbor_radius,
            crs=crs,
            street_view_image=building.get("image"),
            building_mask=building.get("mask"),
            window_mask=building.get("window_mask"),
            door_mask=building.get("door_mask"),
            height_value=building.get("height"),
            material_percentages=building.get("materials"),
        )

        all_properties.append(props)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"✓ Batch complete! Processed {len(buildings)} buildings")
        print(f"{'=' * 60}\n")

    return all_properties


def save_properties_batch(
    properties_list: list[BuildingProperties], output_dir: str = "properties", format: str = "json"
):
    """
    Save a batch of properties to files.

    Parameters
    ----------
    properties_list
        List of BuildingProperties objects.
    output_dir
        Directory to save files.
    format
        'json' or 'csv'.

    Examples
    --------
    >>> save_properties_batch(all_props, "output/", format="json")
    >>> save_properties_batch(all_props, "output/", format="csv")
    """
    import os

    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    if format == "json":
        for props in properties_list:
            filepath = os.path.join(output_dir, f"{props.building_id}.json")
            props.to_json(filepath)
        print(f"✓ Saved {len(properties_list)} JSON files to {output_dir}/")

    elif format == "csv":
        # Create DataFrame
        property_dicts = [p.to_dict() for p in properties_list]
        df = pd.DataFrame(property_dicts)

        # Save
        filepath = os.path.join(output_dir, "all_properties.csv")
        df.to_csv(filepath, index=False)
        print(f"✓ Saved properties to {filepath}")

    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'")


def extract_from_image_path(
    building_id: str, polygon: Polygon, image_path: str, mask_path: str | None = None, **kwargs
) -> BuildingProperties:
    """
    Convenience function when image is a file path instead of array.

    Parameters
    ----------
    building_id
        Building identifier.
    polygon
        Building footprint polygon.
    image_path
        Path to street view image.
    mask_path
        Path to building mask image. Optional.
    **kwargs
        Additional arguments for extract_building_properties.

    Returns
    -------
    BuildingProperties
        Extracted properties.
    """
    from PIL import Image

    # Load image
    image = np.array(Image.open(image_path))

    # Load mask if provided
    mask = None
    if mask_path:
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel if RGB
        mask = mask > 0  # Ensure binary

    return extract_building_properties(
        building_id=building_id, polygon=polygon, street_view_image=image, building_mask=mask, **kwargs
    )
