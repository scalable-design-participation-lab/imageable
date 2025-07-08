from dataclasses import asdict, dataclass
from typing import Any

from imageable.images.camera import CameraParameters


@dataclass
class ImageMetadata:
    """
    Metadata for a Street View image.

    Parameters
    ----------
    status
        Status of the image, True if the image was successfully fetched.
    date
        Date when the image was updated/uploaded to the cloud service.
    img_size
        Size of the image as a tuple (width, height).
    source
        Source or copyright information for the image. The default source will be Google Street View.
    latitude
        Latitude where the image was captured.
    longitude
        Longitude where the image was captured.
    pano_id
        Panorama ID associated with the image within the cloud service.
    camera_parameters
        Camera parameters used to capture the image, including latitude, longitude, heading, pitch,
        fov, width, and height. This can be a CameraParameters object or a dictionary with the same keys.
    """

    # Status of the image
    status: bool
    # Date when the image was updated/uploaded
    date: str
    # size of the image
    img_size: tuple[int, int]
    # source/copyright
    source: str
    # Latitude
    latitude: float
    # longitude
    longitude: float
    # panorama id
    pano_id: str
    # camera parameters
    camera_parameters: CameraParameters | dict

    def __post_init__(self) -> None:
        """Convert camera parameters to a dictionary if necessary."""
        #The default behavior is to convert the camera parameters to a dictionary
        if isinstance(self.camera_parameters, CameraParameters):
            self.camera_parameters = self.camera_parameters.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Convert ImageMetadata to a dictionary."""
        return asdict(self)
