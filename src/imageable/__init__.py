"""
imageable: Computer Vision Library for Urban Building Analysis

Extract 43+ building properties from street view images and footprints.
"""

# Core API
from imageable.core.properties import BuildingProperties
from imageable.core.dataset import get_dataset
from imageable.core.image import get_image

# Version
from imageable._version import __version__

__all__ = [
    # Core
    "BuildingProperties",
    "get_dataset",
    "get_image",
    # Meta
    "__version__",
]

# TODO: Add batch, viz when ready
# from imageable.batch import BatchProcessor
# from imageable.viz import plot_building_properties, create_3d_map