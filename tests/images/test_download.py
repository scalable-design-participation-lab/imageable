"""Tests for the download module."""

import io
from http import HTTPStatus
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from imageable._images.camera.camera_parameters import CameraParameters
from imageable._images.download import fetch_image
from imageable._images.image import ImageMetadata


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_camera_params():
    """Create mock camera parameters."""
    return CameraParameters(
        latitude=40.7128,
        longitude=-74.0060,
        heading=0.0,
        pitch=0.0,
        fov=90.0,
        width=640,
        height=640,
    )


@pytest.fixture
def mock_metadata_response():
    """Create a mock successful metadata response."""
    mock = Mock()
    mock.status_code = HTTPStatus.OK
    mock.json.return_value = {
        "date": "2023-10",
        "copyright": "Google",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "pano_id": "12345",
    }
    return mock


@pytest.fixture
def mock_image_response():
    """Create a mock successful image response."""
    mock = Mock()
    mock.status_code = HTTPStatus.OK
    
    # Create a fake image
    fake_image = np.zeros((10, 10, 3), dtype=np.uint8)
    image_bytes = io.BytesIO()
    Image.fromarray(fake_image).save(image_bytes, format="JPEG")
    image_bytes.seek(0)
    mock.content = image_bytes.read()
    
    return mock


# =============================================================================
# Tests for fetch_image with CameraParameters
# =============================================================================


class TestFetchImage:
    """Tests for the fetch_image function."""

    @patch("imageable._images.download.requests.get")
    def test_fetch_image_success(
        self, mock_requests_get, mock_camera_params, mock_metadata_response, mock_image_response
    ):
        """Test successful image fetch."""
        mock_requests_get.side_effect = [mock_metadata_response, mock_image_response]

        image, metadata = fetch_image("FAKE_KEY", mock_camera_params, save_path=None)

        assert image is not None
        assert isinstance(metadata, ImageMetadata)
        assert metadata.status is True
        assert metadata.latitude == 40.7128
        assert metadata.longitude == -74.0060
        assert metadata.pano_id == "12345"
        assert image.shape == (10, 10, 3)
        assert isinstance(image, np.ndarray)

    @patch("imageable._images.download.requests.get")
    def test_fetch_image_metadata_failure(self, mock_requests_get, mock_camera_params):
        """Test image fetch when metadata request fails."""
        mock_response = Mock()
        mock_response.status_code = HTTPStatus.BAD_REQUEST
        mock_requests_get.return_value = mock_response

        image, metadata = fetch_image("FAKE_KEY", mock_camera_params, save_path=None)

        assert image is None
        assert metadata.status is False

    @patch("imageable._images.download.requests.get")
    def test_fetch_image_image_request_failure(
        self, mock_requests_get, mock_camera_params, mock_metadata_response
    ):
        """Test when metadata succeeds but image request fails."""
        mock_image_fail = Mock()
        mock_image_fail.status_code = HTTPStatus.INTERNAL_SERVER_ERROR

        mock_requests_get.side_effect = [mock_metadata_response, mock_image_fail]

        image, metadata = fetch_image("FAKE_KEY", mock_camera_params, save_path=None)

        assert image is None
        assert metadata.status is False


# =============================================================================
# Tests for CameraParameters validation
# =============================================================================


class TestCameraParametersValidation:
    """Tests for CameraParameters validation."""

    def test_valid_parameters(self):
        """Test that valid parameters work."""
        params = CameraParameters(
            latitude=40.7128,
            longitude=-74.0060,
            heading=90.0,
            pitch=10.0,
            fov=90.0,
            width=640,
            height=640,
        )
        assert params.latitude == 40.7128
        assert params.fov == 90.0

    def test_width_too_large_raises(self):
        """Test that width > 640 raises ValueError."""
        with pytest.raises(ValueError, match="width cannot be greater than 640"):
            CameraParameters(latitude=0, longitude=0, width=641)

    def test_height_too_large_raises(self):
        """Test that height > 640 raises ValueError."""
        with pytest.raises(ValueError, match="height cannot be greater than 640"):
            CameraParameters(latitude=0, longitude=0, height=641)

    def test_fov_too_small_raises(self):
        """Test that FOV < 10 raises ValueError."""
        with pytest.raises(ValueError, match="FOV should be between"):
            CameraParameters(latitude=0, longitude=0, fov=5)

    def test_fov_too_large_raises(self):
        """Test that FOV > 120 raises ValueError."""
        with pytest.raises(ValueError, match="FOV should be between"):
            CameraParameters(latitude=0, longitude=0, fov=125)

    def test_to_dict(self):
        """Test converting parameters to dict."""
        params = CameraParameters(
            latitude=40.7128,
            longitude=-74.0060,
            heading=90.0,
            pitch=10.0,
            fov=90.0,
        )
        d = params.to_dict()
        
        assert isinstance(d, dict)
        assert d["latitude"] == 40.7128
        assert d["longitude"] == -74.0060
        assert d["heading"] == 90.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
