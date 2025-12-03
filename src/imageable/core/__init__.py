"""
Core public API for imageable.

This module provides the main entry points for building analysis:

- ``get_image``: Fetch street view images for building footprints
- ``get_dataset``: Extract all 43+ building properties
- ``BuildingProperties``: Dataclass containing extracted properties

Examples
--------
>>> from imageable.core import get_image, get_dataset, BuildingProperties
>>> from shapely.geometry import Polygon
>>>
>>> footprint = Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])
>>> image, metadata = get_image(api_key, footprint)
>>> props = get_dataset(api_key, footprint)
"""

# Import directly from internal modules to avoid circular imports
# These imports bypass imageable/__init__.py
from imageable._extraction.building import BuildingProperties
from imageable.core.dataset import get_dataset
from imageable.core.image import get_image

__all__ = [
    "BuildingProperties",
    "get_dataset",
    "get_image",
]