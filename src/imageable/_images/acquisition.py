"""
High-level image acquisition with automatic quality optimization.

This module provides a clean interface for acquiring street view images of buildings
with automatic camera parameter refinement to ensure the full façade is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from shapely.geometry import Polygon

from imageable._images.camera.camera_adjustment import CameraParametersRefiner
from imageable._images.camera.camera_parameters import CameraParameters


@dataclass
class ImageAcquisitionConfig:
    """
    Configuration for building image acquisition.

    Parameters
    ----------
    api_key
        Google Street View API key.
    save_directory
        Directory to save images and metadata. If None, images are not saved.
    save_intermediate
        If True and save_directory is set, save all intermediate images
        during refinement (useful for debugging).
    overwrite
        If True, overwrite existing cached images.
    min_floor_ratio
        Minimum ratio of floor pixels required for a valid image.
        The refinement process adjusts pitch/FOV until this is satisfied.
    min_sky_ratio
        Minimum ratio of sky pixels required for a valid image.
        The refinement process adjusts pitch/FOV until this is satisfied.
    max_refinement_iterations
        Maximum number of images to fetch during refinement.
    confidence_threshold
        Confidence threshold for sky/floor detection model.
    polygon_buffer_constant
        Buffer constant for finding observation points around the building.
    """

    api_key: str
    save_directory: str | Path | None = None
    save_intermediate: bool = False
    overwrite: bool = True
    min_floor_ratio: float = 0.00001
    min_sky_ratio: float = 0.1
    max_refinement_iterations: int = 5
    confidence_threshold: float = 0.1
    polygon_buffer_constant: float = 20


@dataclass
class ImageAcquisitionResult:
    """
    Result of building image acquisition.

    Attributes
    ----------
    image
        The acquired image as a numpy array (H, W, 3) in RGB format.
        None if acquisition failed.
    camera_params
        The camera parameters used to capture the image.
    metadata
        Additional metadata about the acquisition.
    success
        Whether the refinement succeeded in finding a view with
        both sky and ground visible.
    from_cache
        Whether the image was loaded from cache.
    """

    image: NDArray[np.uint8] | None
    camera_params: CameraParameters
    metadata: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    from_cache: bool = False

    @property
    def is_valid(self) -> bool:
        """Check if acquisition produced a valid image."""
        return self.image is not None


def acquire_building_image(
    polygon: Polygon,
    config: ImageAcquisitionConfig,
) -> ImageAcquisitionResult:
    """
    Acquire a street view image of a building with automatic quality optimization.

    This function handles:
    1. Finding the best observation point for the building
    2. Fetching an initial image from Google Street View
    3. Iteratively refining camera pitch and FOV to ensure the full façade is visible
    4. Caching results for subsequent calls

    Parameters
    ----------
    polygon
        Shapely polygon representing the building footprint.
    config
        Configuration for the acquisition process.

    Returns
    -------
    result
        An ImageAcquisitionResult containing the image, camera parameters,
        and metadata about the acquisition.

    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> from imageable._images.acquisition import acquire_building_image, ImageAcquisitionConfig
    >>>
    >>> footprint = Polygon([(-71.05, 42.36), (-71.05, 42.37), (-71.04, 42.37), (-71.04, 42.36)])
    >>> config = ImageAcquisitionConfig(api_key="your_key", save_directory="./images")
    >>> result = acquire_building_image(footprint, config)
    >>> if result.is_valid:
    ...     print(f"Image shape: {result.image.shape}")
    ...     print(f"Camera FOV: {result.camera_params.fov}")
    """
    save_dir = Path(config.save_directory) if config.save_directory else None

    # Check cache first
    if save_dir and not config.overwrite:
        cached_result = _load_from_cache(save_dir)
        if cached_result is not None:
            return cached_result

    # Perform acquisition with refinement
    refiner = CameraParametersRefiner(polygon)
    refiner.MIN_FLOOR_RATIO = config.min_floor_ratio
    refiner.MIN_SKY_RATIO = config.min_sky_ratio

    camera_params, refinement_success, image = refiner.adjust_parameters(
        api_key=config.api_key,
        max_number_of_images=config.max_refinement_iterations,
        polygon_buffer_constant=config.polygon_buffer_constant,
        pictures_directory=str(save_dir) if save_dir else None,
        save_reel=config.save_intermediate,
        overwrite_images=config.overwrite,
        confidence_detection=config.confidence_threshold,
    )

    metadata = {
        "refinement_iterations": config.max_refinement_iterations,
        "refinement_success": refinement_success,
        "min_floor_ratio": config.min_floor_ratio,
        "min_sky_ratio": config.min_sky_ratio,
    }

    return ImageAcquisitionResult(
        image=image,
        camera_params=camera_params,
        metadata=metadata,
        success=refinement_success,
        from_cache=False,
    )


def _load_from_cache(save_dir: Path) -> ImageAcquisitionResult | None:
    """
    Attempt to load a previously acquired image from cache.

    Parameters
    ----------
    save_dir
        Directory where cached images are stored.

    Returns
    -------
    result
        ImageAcquisitionResult if cache hit, None otherwise.
    """
    image_path = save_dir / "image.jpg"
    metadata_path = save_dir / "metadata.json"

    if not (image_path.exists() and metadata_path.exists()):
        return None

    try:
        image = np.array(Image.open(image_path))

        with metadata_path.open("r") as f:
            metadata_dict = json.load(f)

        cam_dict = metadata_dict.get("camera_parameters", {})
        camera_params = CameraParameters(
            longitude=cam_dict["longitude"],
            latitude=cam_dict["latitude"],
            fov=cam_dict.get("fov", 90),
            heading=cam_dict.get("heading", 0),
            pitch=cam_dict.get("pitch", 0),
            width=cam_dict.get("width", 640),
            height=cam_dict.get("height", 640),
        )

        return ImageAcquisitionResult(
            image=image,
            camera_params=camera_params,
            metadata=metadata_dict,
            success=True,  # Assume cached images were successful
            from_cache=True,
        )

    except Exception:
        return None


def load_image_with_metadata(
    image_path: str | Path,
    metadata_path: str | Path | None = None,
) -> ImageAcquisitionResult:
    """
    Load a previously saved image and its metadata.

    This is useful for processing images that were acquired separately
    or for testing with pre-existing images.

    Parameters
    ----------
    image_path
        Path to the image file.
    metadata_path
        Path to the metadata JSON file. If None, attempts to find
        metadata.json in the same directory as the image.

    Returns
    -------
    result
        ImageAcquisitionResult with the loaded image and camera parameters.

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = np.array(Image.open(image_path))

    # Try to load metadata
    if metadata_path is None:
        metadata_path = image_path.parent / "metadata.json"

    camera_params = CameraParameters(
        longitude=0.0,
        latitude=0.0,
        fov=90,
        heading=0,
        pitch=0,
        width=image.shape[1],
        height=image.shape[0],
    )
    metadata = {}

    if Path(metadata_path).exists():
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            cam_dict = metadata.get("camera_parameters", {})
            camera_params = CameraParameters(
                longitude=cam_dict.get("longitude", 0.0),
                latitude=cam_dict.get("latitude", 0.0),
                fov=cam_dict.get("fov", 90),
                heading=cam_dict.get("heading", 0),
                pitch=cam_dict.get("pitch", 0),
                width=cam_dict.get("width", image.shape[1]),
                height=cam_dict.get("height", image.shape[0]),
            )
        except Exception:
            pass  # Use defaults if metadata parsing fails

    return ImageAcquisitionResult(
        image=image,
        camera_params=camera_params,
        metadata=metadata,
        success=True,
        from_cache=True,
    )
