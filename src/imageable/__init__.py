"""
imageable: Computer Vision Library for Urban Building Analysis.

Extract 43+ building properties from street view images and footprints.
Designed for urban planners, architects, and climate resilience research.

Quick Start
-----------
>>> import imageable
>>> from shapely.geometry import Polygon
>>>
>>> # Define building footprint
>>> footprint = Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])
>>>
>>> # Get street view image
>>> image, metadata = imageable.get_image(api_key, footprint)
>>>
>>> # Extract all 43+ properties
>>> props = imageable.get_dataset(api_key, footprint)
>>> print(f"Height: {props.building_height}m")
>>> print(f"Area: {props.projected_area}m²")

Modules
-------
core
    High-level API for property extraction
batch
    Batch processing for GeoDataFrames (coming soon)
viz
    Visualization tools (coming soon)
models
    Advanced access to ML models
"""

from __future__ import annotations


# Version - import first as it has no dependencies
from imageable._version import __version__

from imageable._extraction.building import BuildingProperties
from imageable.core.dataset import get_dataset
from imageable.core.image import get_image

# Common types users might need
from imageable._images.camera.camera_parameters import CameraParameters
from imageable._images.image import ImageMetadata


__all__ = [
    # Meta
    "__version__",
    # Core API
    "BuildingProperties",
    "get_dataset",
    "get_image",
    # Types
    "CameraParameters",
    "ImageMetadata",
]

# Version tuple for programmatic access
__version_info__ = tuple(int(x) for x in __version__.split(".")[:3])

