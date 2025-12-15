"""
imageable: Computer Vision Library for Urban Building Analysis

Extract 43+ building properties from street view images and footprints.
"""

# Core API - explicit input functions
from imageable.core.building_data import (
    get_building_data_from_gdf,
    get_building_data_from_geojson,
    get_building_data_from_file,
)

# Data classes
from imageable._extraction.building import BuildingProperties

# Single building / image utilities
from imageable.core.dataset import get_dataset
from imageable.core.image import get_image

# Version
from imageable._version import __version__

__all__ = [
    # Main API (new)
    "get_building_data_from_gdf",
    "get_building_data_from_geojson", 
    "get_building_data_from_file",
    # Single building
    "get_dataset",
    "get_image",
    # Data classes
    "BuildingProperties",
    # Meta
    "__version__",
]