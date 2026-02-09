"""Tests for core.image module - high-level image acquisition API."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from shapely.geometry import Polygon

from imageable.core.image import get_image, load_image
from imageable._images.camera.camera_parameters import CameraParameters


@pytest.fixture
def simple_polygon():
    """Create a simple rectangular polygon for testing."""
    return Polygon([(0, 0), (10, 0), (10, 20), (0, 20)])


@pytest.fixture
def mock_image():
    """Create a mock image array."""
    return np.zeros((512, 512, 3), dtype=np.uint8)


@pytest.fixture
def mock_camera_params():
    """Create mock camera parameters."""
    return CameraParameters(
        longitude=-71.0589,
        latitude=42.3601,
        fov=90,
        heading=45,
        pitch=10,
        width=512,
        height=512,
    )


class TestGetImage:
    """Tests for the get_image function."""

    @patch("imageable.core.image.acquire_building_image")
    def test_returns_image_and_metadata(
        self, mock_acquire, simple_polygon, mock_image, mock_camera_params
    ):
        """Test that get_image returns image, camera_params and metadata tuple."""
        from imageable._images.acquisition import ImageAcquisitionResult
        
        mock_acquire.return_value = ImageAcquisitionResult(
            image=mock_image,
            camera_params=mock_camera_params,
            metadata={"refinement_success": True},
            success=True,
        )

        image, camera_params, metadata = get_image("api_key", simple_polygon)

        assert isinstance(image, np.ndarray)
        assert image.shape == (512, 512, 3)
        assert camera_params.fov == 90
        assert isinstance(metadata, dict)

    @patch("imageable.core.image.acquire_building_image")
    def test_returns_only_image_when_metadata_disabled(
        self, mock_acquire, simple_polygon, mock_image, mock_camera_params
    ):
        """Test that only image is returned when return_metadata=False."""
        from imageable._images.acquisition import ImageAcquisitionResult
        
        mock_acquire.return_value = ImageAcquisitionResult(
            image=mock_image,
            camera_params=mock_camera_params,
            metadata={},
            success=True,
        )

        result = get_image("api_key", simple_polygon, return_metadata=False)

        assert isinstance(result, np.ndarray)
        assert result.shape == (512, 512, 3)

    @patch("imageable.core.image.acquire_building_image")
    def test_passes_save_path(self, mock_acquire, simple_polygon, mock_image, mock_camera_params, tmp_path):
        """Test that save_path is passed to acquisition config."""
        from imageable._images.acquisition import ImageAcquisitionResult
        
        mock_acquire.return_value = ImageAcquisitionResult(
            image=mock_image,
            camera_params=mock_camera_params,
            metadata={},
            success=True,
        )
        save_path = tmp_path

        get_image("api_key", simple_polygon, save_path=save_path)

        mock_acquire.assert_called_once()
        # Check that the config has the save directory
        call_args = mock_acquire.call_args
        config = call_args[1]["config"]
        assert config.save_directory == str(save_path)

    @patch("imageable.core.image.acquire_building_image")
    def test_passes_api_key_and_polygon(
        self, mock_acquire, simple_polygon, mock_image, mock_camera_params
    ):
        """Test that API key and polygon are passed correctly."""
        from imageable._images.acquisition import ImageAcquisitionResult
        
        mock_acquire.return_value = ImageAcquisitionResult(
            image=mock_image,
            camera_params=mock_camera_params,
            metadata={},
            success=True,
        )

        get_image("my_api_key", simple_polygon)

        mock_acquire.assert_called_once()
        call_args = mock_acquire.call_args
        # First positional arg is polygon
        assert call_args[0][0] is simple_polygon
        # Config should have the api key
        config = call_args[1]["config"]
        assert config.api_key == "my_api_key"

    @patch("imageable.core.image.acquire_building_image")
    def test_refine_camera_disabled(
        self, mock_acquire, simple_polygon, mock_image, mock_camera_params
    ):
        """Test that refinement can be disabled."""
        from imageable._images.acquisition import ImageAcquisitionResult
        
        mock_acquire.return_value = ImageAcquisitionResult(
            image=mock_image,
            camera_params=mock_camera_params,
            metadata={},
            success=True,
        )

        get_image("api_key", simple_polygon, refine_camera=False)

        config = mock_acquire.call_args[1]["config"]
        # When refine_camera=False, iterations should be 1
        assert config.max_refinement_iterations == 1

    @patch("imageable.core.image.acquire_building_image")
    def test_raises_on_failed_acquisition(
        self, mock_acquire, simple_polygon
    ):
        """Test that RuntimeError is raised when acquisition fails."""
        from imageable._images.acquisition import ImageAcquisitionResult
        
        mock_acquire.return_value = ImageAcquisitionResult(
            image=None,
            camera_params=CameraParameters(longitude=0, latitude=0),
            metadata={},
            success=False,
        )

        with pytest.raises(RuntimeError, match="Failed to acquire image"):
            get_image("api_key", simple_polygon)


class TestLoadImage:
    """Tests for the load_image function."""

    def test_load_existing_image(self, tmp_path, mock_image):
        """Test loading an existing image."""
        import json
        from PIL import Image
        
        # Create test image
        image_path = tmp_path / "image.jpg"
        Image.fromarray(mock_image).save(image_path)
        
        # Create metadata
        metadata_path = tmp_path / "metadata.json"
        metadata = {"camera_parameters": {"fov": 90, "longitude": -71.0, "latitude": 42.0}}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        image, camera_params, meta = load_image(image_path)
        
        assert image.shape == (512, 512, 3)
        assert camera_params.fov == 90

    def test_load_nonexistent_raises(self, tmp_path):
        """Test that loading nonexistent image raises error."""
        with pytest.raises(RuntimeError, match="Failed to load image"):
            load_image(tmp_path / "nonexistent.jpg")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
