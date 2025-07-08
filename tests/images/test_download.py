import io
from http import HTTPStatus
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from imageable.images.camera import CameraParameters
from imageable.images.download import fetch_image
from imageable.images.image import ImageMetadata


@patch("imageable.images.download.requests.get")

def test_fetch_image_success(mock_requests_get):
    mock_metadata_response = Mock()
    mock_metadata_response.status_code = HTTPStatus.OK

    #The simulated metadata response
    mock_metadata_response.json.return_value = {
        "date": "2023-10",
        "copyright":"Uriel simulations",
        "location":{"lat": 40.7128, "lng": -74.0060},
        "pano_id":"12345"
    }

    #The simulated image response
    mock_image_response = Mock()
    mock_image_response.status_code = 200
    fake_image = np.zeros((10,10,3), dtype = np.uint8)
    image_bytes = io.BytesIO()
    Image.fromarray(fake_image).save(image_bytes,format = "JPEG")
    image_bytes.seek(0)
    mock_image_response.content = image_bytes.read()

    #Assign mocks to call sequence
    mock_requests_get.side_effect = [mock_metadata_response, mock_image_response]

    #Dummy camera parameters
    camera_parameters = CameraParameters(
        latitude=40.7128,
        longitude=-74.0060,
        heading = 0.0,
        pitch = 0.0,
        fov = 90.0,
        width=640,
        height=640
    )
    #Call the function to test

    image,metadata = fetch_image(
        "FAKE_KEY",
        camera_parameters,
        save_path = None
    )

    #Verify the image and metadata
    assert image is not None
    assert isinstance(metadata, ImageMetadata)
    assert metadata.status
    assert metadata.latitude == 40.7128
    assert metadata.longitude == -74.0060
    assert metadata.pano_id == "12345"
    assert image.shape == (10, 10, 3)
    assert isinstance(image, np.ndarray)





