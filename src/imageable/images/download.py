import io
import json
import logging
from http import HTTPStatus
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import requests
from PIL import Image

from imageable.images.camera import CameraParameters
from imageable.images.image import ImageMetadata

NA_FIELD = "N/A"
RESPONSE_TIMEOUT = 10


def _save_metadata(save_path: str, metadata_dictionary: dict[str, Any]) -> None:
    """Save metadata to a JSON file."""
    try:
        parent_path = Path(save_path).parent
        # Ensure the directory exists
        parent_path.mkdir(parents = True, exist_ok = True)
        metadata_path = parent_path / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata_dictionary, f, indent=4)
    except Exception:
        logging.exception("Failed to save metadata")


def fetch_image(
    api_key: str,
    camera_parameters: CameraParameters,
    save_path: str | None,
    overwrite_image: bool =True
) -> tuple[npt.NDArray[np.uint8]|None, ImageMetadata|None]:
    """
    Interface applied to fetch the Street View image based on CameraParameters.

    Parameters
    ----------
    api_key
        Google API key.
    camera_parameters
        The paremeters applied to fetch the image.
    save_path
        If None the image will not be saved, otherwise the image will be saved to the specified path.
    overwrite_image
        If True and save_path is not None, the image will be overwritten if it already exists.

    Returns
    -------
    _______
        image
            The requested image as a numpy array
        metadata
            Image metadata
    """
    # Check if the image already exists and should not be overwritten
    if save_path is not None and not overwrite_image and Path(save_path).exists():
        return None, None
    # Base URL for the Google Street View API
    base_url = "https://maps.googleapis.com/maps/api/streetview"

    # Construct the parameters for the API request
    params = {
        "location": f"{camera_parameters.latitude}, {camera_parameters.longitude}",
        "heading": str(camera_parameters.heading),
        "pitch": str(camera_parameters.pitch),
        "fov": str(camera_parameters.fov),
        "size": f"{camera_parameters.width}x{camera_parameters.height}",
        "key": api_key,
    }

    # First let's fetch the metadata
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    response_metadata = requests.get(metadata_url, params=params, timeout = RESPONSE_TIMEOUT)

    # If the status code is 200, we can extract the metadata
    if response_metadata.status_code == HTTPStatus.OK:
        metadata = response_metadata.json()
        # Fill the fields of the metadata
        status = True
        date = metadata.get("date", NA_FIELD)
        imgsize = (camera_parameters.width, camera_parameters.height)
        source = metadata.get("copyright", NA_FIELD)
        latitude = metadata.get("location", {}).get("lat", np.nan)
        longitude = metadata.get("location", {}).get("lng", np.nan)
        pano_id = metadata.get("pano_id", NA_FIELD)

        img_metadata = ImageMetadata(
            status=status,
            date=date,
            img_size=imgsize,
            source=source,
            latitude=latitude,
            longitude=longitude,
            pano_id=pano_id,
            camera_parameters=camera_parameters,
        )

        # Now let's get the image
        response = requests.get(base_url, params=params, stream=True, timeout=RESPONSE_TIMEOUT)
        image_array = None

        if response.status_code == HTTPStatus.OK:
            image = response.content
            image = io.BytesIO(image)
            image_array = np.array(Image.open(image))
            if save_path is not None:
                img_save_path = Path(save_path)
                img_save_path.parent.mkdir(parents = True, exist_ok = True)
                with img_save_path.open("wb") as f:
                    f.write(response.content)
        else:
            img_metadata.status = False

        # If save_path is not None, we will save the metadata
        if save_path is not None:
            _save_metadata(save_path, img_metadata.to_dict())
        return image_array, img_metadata

    # We will return an empty metadata in cases where request fails
    img_metadata = ImageMetadata(
        status=False,
        date=NA_FIELD,
        img_size=(None, None),
        source=NA_FIELD,
        latitude=np.nan,
        longitude=np.nan,
        pano_id=NA_FIELD,
        camera_parameters=camera_parameters.to_dict(),
    )
    if save_path is not None:
        _save_metadata(save_path, img_metadata.to_dict())
    return None, img_metadata
