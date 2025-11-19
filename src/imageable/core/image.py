"""High-level image acquisition API."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon

from imageable._images.download import download_street_view_image 


def get_image(
    key: str,
    building_footprint: Polygon,
    *,
    save_path: str | Path | None = None,
    return_metadata: bool = True,
) -> tuple[NDArray, dict[str, Any]] | NDArray:
    """
    Get street view image for a building footprint.

    Parameters
    ----------
    key : str
        Google Street View API key.
    building_footprint : shapely.Polygon
        Building footprint geometry.
    save_path : str or Path, optional
        Path to save the downloaded image.
    return_metadata : bool, default=True
        Whether to return camera metadata.

    Returns
    -------
    image : ndarray
        Street view image as numpy array (H, W, 3).
    metadata : dict, optional
        Camera parameters and metadata (if return_metadata=True).

    Examples
    --------
    >>> from imageable import get_image
    >>> from shapely.geometry import Polygon
    >>> 
    >>> footprint = Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])
    >>> image, metadata = get_image(api_key, footprint)
    >>> image.shape
    (512, 512, 3)
    """
    # Use existing download function
    result = download_street_view_image(
        api_key=key,
        building_polygon=building_footprint,
        save_path=str(save_path) if save_path else None,
    )
    
    if return_metadata:
        return result["image"], result["metadata"]
    else:
        return result["image"]