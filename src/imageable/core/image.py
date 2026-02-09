"""
High-level image acquisition API.

This module provides standalone functions for acquiring street view images of buildings.
These functions work independently of any analysis pipeline and can be used for:

- Collecting images for custom computer vision tasks
- Building image datasets for training ML models
- Visual inspection and quality control
- Pre-acquisition before analysis to save API costs
- Creating image libraries for urban research

The image acquisition system automatically:
- Finds optimal observation points using street networks
- Refines camera parameters (pitch, FOV) to capture full building façades
- Caches results to avoid redundant API calls
- Validates image quality (sky and ground visibility)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from imageable._images.acquisition import (
    ImageAcquisitionConfig,
    ImageAcquisitionResult,
    acquire_building_image,
    load_image_with_metadata,
)
from imageable._images.camera.camera_parameters import CameraParameters


def get_image(
    key: str,
    building_footprint: Polygon,
    *,
    save_path: str | Path | None = None,
    return_metadata: bool = True,
    refine_camera: bool = True,
    min_floor_ratio: float = 0.00001,
    min_sky_ratio: float = 0.1,
    max_refinement_iterations: int = 5,
    overwrite: bool = True,
) -> tuple[NDArray, CameraParameters] | tuple[NDArray, CameraParameters, dict[str, Any]] | NDArray:
    """
    Get street view image for a building footprint.

    This function acquires a street view image of a building, optionally
    refining camera parameters to ensure the full façade is visible.

    Parameters
    ----------
    key : str
        Google Street View API key.
    building_footprint : shapely.Polygon
        Building footprint geometry.
    save_path : str or Path, optional
        Directory to save the downloaded image and metadata.
    return_metadata : bool, default=True
        Whether to return camera parameters and additional metadata.
    refine_camera : bool, default=True
        Whether to refine camera pitch/FOV to ensure sky and ground are visible.
        Set to False for faster acquisition without quality optimization.
    min_floor_ratio : float, default=0.00001
        Minimum ratio of floor pixels for refinement.
    min_sky_ratio : float, default=0.1
        Minimum ratio of sky pixels for refinement.
    max_refinement_iterations : int, default=5
        Maximum number of images to fetch during refinement.
    overwrite : bool, default=True
        Whether to overwrite existing cached images.

    Returns
    -------
    image : ndarray
        Street view image as numpy array (H, W, 3) in RGB format.
    camera_params : CameraParameters
        Camera parameters used to capture the image (if return_metadata=True).
    metadata : dict, optional
        Additional metadata about the acquisition.

    Examples
    --------
    Basic usage - get an image with automatic refinement:
    
    >>> from imageable import get_image
    >>> from shapely.geometry import Polygon
    >>> 
    >>> footprint = Polygon([(-71.05, 42.36), (-71.05, 42.37), (-71.04, 42.37), (-71.04, 42.36)])
    >>> image, camera_params, metadata = get_image(api_key, footprint)
    >>> image.shape
    (640, 640, 3)
    >>> camera_params.fov  # Automatically adjusted to show full façade
    90
    
    Fast acquisition without refinement (when you just need any image):
    
    >>> image = get_image(api_key, footprint, refine_camera=False, return_metadata=False)
    >>> # Single API call, no quality optimization
    
    Save images for later analysis:
    
    >>> image, params, meta = get_image(
    ...     api_key, 
    ...     footprint,
    ...     save_path="./building_images/building_001"
    ... )
    >>> # Image and metadata saved to disk, can be loaded later with load_image()
    
    Collect multiple images with custom quality thresholds:
    
    >>> images = []
    >>> for footprint in building_footprints:
    ...     img, params = get_image(
    ...         api_key,
    ...         footprint,
    ...         min_floor_ratio=0.001,  # More lenient
    ...         min_sky_ratio=0.2,      # Require more sky
    ...         max_refinement_iterations=10  # More attempts
    ...     )
    ...     images.append(img)
    """
    config = ImageAcquisitionConfig(
        api_key=key,
        save_directory=str(save_path) if save_path else None,
        overwrite=overwrite,
        min_floor_ratio=min_floor_ratio if refine_camera else 0.0,
        min_sky_ratio=min_sky_ratio if refine_camera else 0.0,
        max_refinement_iterations=max_refinement_iterations if refine_camera else 1,
    )

    result = acquire_building_image(building_footprint, config)

    if not result.is_valid:
        raise RuntimeError("Failed to acquire image for building footprint")

    if return_metadata:
        return result.image, result.camera_params, result.metadata
    else:
        return result.image


def load_image(
    image_path: str | Path,
    metadata_path: str | Path | None = None,
) -> tuple[NDArray, CameraParameters, dict[str, Any]]:
    """
    Load a previously saved building image with its metadata.

    Parameters
    ----------
    image_path : str or Path
        Path to the image file.
    metadata_path : str or Path, optional
        Path to the metadata JSON file. If None, looks for metadata.json
        in the same directory as the image.

    Returns
    -------
    image : ndarray
        Image as numpy array (H, W, 3).
    camera_params : CameraParameters
        Camera parameters from metadata.
    metadata : dict
        Full metadata dictionary.

    Examples
    --------
    Load a previously acquired image:
    
    >>> from imageable import load_image
    >>> image, camera_params, metadata = load_image("./images/building_1/image.jpg")
    >>> print(f"Camera was at: ({camera_params.latitude}, {camera_params.longitude})")
    >>> print(f"FOV: {camera_params.fov}°, Pitch: {camera_params.pitch}°")
    
    Load with explicit metadata path:
    
    >>> image, params, meta = load_image(
    ...     "./images/img_001.jpg",
    ...     metadata_path="./metadata/img_001_meta.json"
    ... )
    
    Batch load multiple images:
    
    >>> from pathlib import Path
    >>> image_dir = Path("./building_dataset")
    >>> dataset = []
    >>> for img_path in image_dir.glob("*/image.jpg"):
    ...     img, params, meta = load_image(img_path)
    ...     dataset.append((img, params, meta))
    """
    result = load_image_with_metadata(image_path, metadata_path)

    if not result.is_valid:
        raise RuntimeError(f"Failed to load image from {image_path}")

    return result.image, result.camera_params, result.metadata


# Re-export acquisition types for advanced users
__all__ = [
    "get_image",
    "load_image",
    "ImageAcquisitionConfig",
    "ImageAcquisitionResult",
    "acquire_building_image",
    "CameraParameters",
]
