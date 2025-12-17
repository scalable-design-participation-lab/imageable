"""High-level dataset extraction API."""

from typing import Any
from shapely.geometry import Polygon

from imageable._extraction.building import BuildingProperties
from imageable._extraction.extract import extract_building_properties


def get_dataset(
    key: str,
    building_footprint: Polygon,
    *,
    neighbor_radius: float = 100.0,
    crs: int = 4326,
    image: Any = None,
    verbose: bool = False,
) -> BuildingProperties:
    """
    Extract comprehensive building dataset from footprint and image.

    Parameters
    ----------
    key : str
        Google Street View API key.
    building_footprint : shapely.Polygon
        Building footprint geometry.
    neighbor_radius : float, default=100.0
        Radius for neighbor analysis (meters).
    crs : str, default="EPSG:4326"
        Coordinate reference system.
    image : ndarray, optional
        Pre-fetched street view image.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    BuildingProperties
        Dataclass containing all 43+ extracted properties.

    Examples
    --------
    >>> from imageable import get_dataset
    >>> from shapely.geometry import Polygon
    >>> 
    >>> footprint = Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])
    >>> props = get_dataset(api_key, footprint)
    >>> print(f"Height: {props.building_height}m")
    """
    # Use existing extract function
    return extract_building_properties(
        building_id=key, 
        polygon=building_footprint,
        all_buildings=[],  # Single building mode
        neighbor_radius=neighbor_radius,
        crs=crs,
        street_view_image=image,
    )