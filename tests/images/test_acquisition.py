"""Tests for the image acquisition module."""

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from shapely.geometry import Polygon

from imageable._images.acquisition import (
    ImageAcquisitionConfig,
    ImageAcquisitionResult,
    acquire_building_image,
    load_image_with_metadata,
    _load_from_cache,
)
from imageable._images.camera.camera_parameters import CameraParameters


class TestImageAcquisitionConfig:
    """Tests for ImageAcquisitionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ImageAcquisitionConfig(api_key="test_key")
        
        assert config.api_key == "test_key"
        assert config.save_directory is None
        assert config.save_intermediate is False
        assert config.overwrite is True
        assert config.min_floor_ratio == 0.00001
        assert config.min_sky_ratio == 0.1
        assert config.max_refinement_iterations == 5
        assert config.confidence_threshold == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ImageAcquisitionConfig(
            api_key="my_key",
            save_directory="/tmp/images",
            save_intermediate=True,
            overwrite=False,
            min_floor_ratio=0.001,
            min_sky_ratio=0.2,
            max_refinement_iterations=10,
        )
        
        assert config.api_key == "my_key"
        assert config.save_directory == "/tmp/images"
        assert config.save_intermediate is True
        assert config.overwrite is False
        assert config.min_floor_ratio == 0.001
        assert config.min_sky_ratio == 0.2
        assert config.max_refinement_iterations == 10


class TestImageAcquisitionResult:
    """Tests for ImageAcquisitionResult dataclass."""

    @pytest.fixture
    def sample_camera_params(self):
        """Create sample camera parameters."""
        return CameraParameters(
            longitude=-71.05,
            latitude=42.36,
            fov=90,
            heading=45,
            pitch=10,
            width=640,
            height=640,
        )

    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def test_valid_result(self, sample_image, sample_camera_params):
        """Test a valid acquisition result."""
        result = ImageAcquisitionResult(
            image=sample_image,
            camera_params=sample_camera_params,
            success=True,
        )
        
        assert result.is_valid is True
        assert result.success is True
        assert result.from_cache is False
        assert result.image.shape == (640, 640, 3)

    def test_invalid_result(self, sample_camera_params):
        """Test an invalid acquisition result (no image)."""
        result = ImageAcquisitionResult(
            image=None,
            camera_params=sample_camera_params,
            success=False,
        )
        
        assert result.is_valid is False
        assert result.success is False

    def test_cached_result(self, sample_image, sample_camera_params):
        """Test a cached result."""
        result = ImageAcquisitionResult(
            image=sample_image,
            camera_params=sample_camera_params,
            success=True,
            from_cache=True,
        )
        
        assert result.is_valid is True
        assert result.from_cache is True


class TestAcquireBuildingImage:
    """Tests for acquire_building_image function."""

    @pytest.fixture
    def sample_polygon(self):
        """Create a sample building polygon."""
        return Polygon([
            (-71.05, 42.36),
            (-71.05, 42.37),
            (-71.04, 42.37),
            (-71.04, 42.36),
        ])

    @pytest.fixture
    def mock_refiner(self):
        """Create a mock CameraParametersRefiner."""
        with patch("imageable._images.acquisition.CameraParametersRefiner") as mock:
            mock_instance = Mock()
            mock.return_value = mock_instance
            
            # Configure mock to return successful result
            mock_camera = CameraParameters(
                longitude=-71.05,
                latitude=42.36,
                fov=90,
                heading=45,
                pitch=10,
                width=640,
                height=640,
            )
            mock_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            mock_instance.adjust_parameters.return_value = (mock_camera, True, mock_image)
            
            yield mock_instance

    def test_successful_acquisition(self, sample_polygon, mock_refiner):
        """Test successful image acquisition."""
        config = ImageAcquisitionConfig(api_key="test_key")
        
        result = acquire_building_image(sample_polygon, config)
        
        assert result.is_valid is True
        assert result.success is True
        assert result.from_cache is False
        mock_refiner.adjust_parameters.assert_called_once()

    def test_acquisition_with_save_directory(self, sample_polygon, mock_refiner, tmp_path):
        """Test acquisition with save directory."""
        config = ImageAcquisitionConfig(
            api_key="test_key",
            save_directory=str(tmp_path),
        )
        
        result = acquire_building_image(sample_polygon, config)
        
        assert result.is_valid is True
        # Verify refiner was called with pictures_directory
        call_kwargs = mock_refiner.adjust_parameters.call_args[1]
        assert call_kwargs["pictures_directory"] == str(tmp_path)

    def test_acquisition_parameters_passed_to_refiner(self, sample_polygon, mock_refiner):
        """Test that config parameters are correctly passed to refiner."""
        config = ImageAcquisitionConfig(
            api_key="test_key",
            min_floor_ratio=0.01,
            min_sky_ratio=0.2,
            max_refinement_iterations=10,
            confidence_threshold=0.5,
        )
        
        acquire_building_image(sample_polygon, config)
        
        # Check that ratios were set on refiner
        assert mock_refiner.MIN_FLOOR_RATIO == 0.01
        assert mock_refiner.MIN_SKY_RATIO == 0.2
        
        # Check call arguments
        call_kwargs = mock_refiner.adjust_parameters.call_args[1]
        assert call_kwargs["max_number_of_images"] == 10
        assert call_kwargs["confidence_detection"] == 0.5

    def test_failed_acquisition(self, sample_polygon, mock_refiner):
        """Test handling of failed acquisition."""
        mock_refiner.adjust_parameters.return_value = (
            CameraParameters(longitude=0, latitude=0),
            False,
            None,
        )
        
        config = ImageAcquisitionConfig(api_key="test_key")
        result = acquire_building_image(sample_polygon, config)
        
        assert result.is_valid is False
        assert result.success is False


class TestLoadFromCache:
    """Tests for cache loading functionality."""

    def test_cache_hit(self, tmp_path):
        """Test loading from cache when files exist."""
        import json
        from PIL import Image
        
        # Create test image
        image_path = tmp_path / "image.jpg"
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        Image.fromarray(test_image).save(image_path)
        
        # Create test metadata
        metadata_path = tmp_path / "metadata.json"
        metadata = {
            "camera_parameters": {
                "longitude": -71.05,
                "latitude": 42.36,
                "fov": 90,
                "heading": 45,
                "pitch": 10,
                "width": 640,
                "height": 640,
            }
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        result = _load_from_cache(tmp_path)
        
        assert result is not None
        assert result.is_valid is True
        assert result.from_cache is True
        assert result.camera_params.fov == 90
        assert result.camera_params.heading == 45

    def test_cache_miss_no_files(self, tmp_path):
        """Test cache miss when files don't exist."""
        result = _load_from_cache(tmp_path)
        
        assert result is None

    def test_cache_miss_partial_files(self, tmp_path):
        """Test cache miss when only image exists."""
        from PIL import Image
        
        image_path = tmp_path / "image.jpg"
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image).save(image_path)
        
        result = _load_from_cache(tmp_path)
        
        assert result is None


class TestLoadImageWithMetadata:
    """Tests for load_image_with_metadata function."""

    def test_load_with_metadata(self, tmp_path):
        """Test loading image with metadata file."""
        import json
        from PIL import Image
        
        # Create test image
        image_path = tmp_path / "image.jpg"
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(test_image).save(image_path)
        
        # Create metadata
        metadata_path = tmp_path / "metadata.json"
        metadata = {
            "camera_parameters": {
                "longitude": -71.05,
                "latitude": 42.36,
                "fov": 90,
            }
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        result = load_image_with_metadata(image_path)
        
        assert result.is_valid is True
        assert result.image.shape == (480, 640, 3)
        assert result.camera_params.fov == 90

    def test_load_without_metadata(self, tmp_path):
        """Test loading image without metadata file (uses defaults)."""
        from PIL import Image
        
        image_path = tmp_path / "image.jpg"
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        Image.fromarray(test_image).save(image_path)
        
        result = load_image_with_metadata(image_path)
        
        assert result.is_valid is True
        assert result.image.shape == (480, 640, 3)
        # Should use default camera params
        assert result.camera_params.fov == 90

    def test_load_nonexistent_image(self, tmp_path):
        """Test loading nonexistent image raises error."""
        with pytest.raises(FileNotFoundError):
            load_image_with_metadata(tmp_path / "nonexistent.jpg")

    def test_load_with_explicit_metadata_path(self, tmp_path):
        """Test loading with explicitly specified metadata path."""
        import json
        from PIL import Image
        
        # Create image in one location
        image_path = tmp_path / "images" / "test.jpg"
        image_path.parent.mkdir(parents=True)
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image).save(image_path)
        
        # Create metadata in different location
        metadata_path = tmp_path / "meta" / "info.json"
        metadata_path.parent.mkdir(parents=True)
        with open(metadata_path, "w") as f:
            json.dump({"camera_parameters": {"fov": 120}}, f)
        
        result = load_image_with_metadata(image_path, metadata_path)
        
        assert result.camera_params.fov == 120
