"""Tests for building height estimation module."""

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
from shapely import Polygon

from imageable._features.height.building_height import (
    HeightEstimationParameters,
    HeightEstimationConfig,
    building_height_from_single_view,
    estimate_height_from_image,
    collect_heights,
    mean_no_outliers,
)
from imageable._images.camera.camera_parameters import CameraParameters


class TestHeightEstimationParameters:
    """Test suite for HeightEstimationParameters dataclass."""

    def test_initialization_with_required_params(self):
        """Test initialization with only required parameters."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        params = HeightEstimationParameters(gsv_api_key="test_api_key", building_polygon=polygon)

        assert params.gsv_api_key == "test_api_key"
        assert params.building_polygon == polygon
        assert params.confidence_detection == 0.1
        assert params.max_number_of_images == 5
        assert params.verbose is False

    def test_initialization_with_custom_values(self):
        """Test initialization with custom parameter values."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        params = HeightEstimationParameters(
            gsv_api_key="custom_key",
            building_polygon=polygon,
            confidence_detection=0.2,
            max_number_of_images=10,
            min_floor_ratio=0.0001,
            min_sky_ratio=0.15,
            device_seg="cuda",
            device_lcnn="cuda",
            verbose=True,
            use_pitch_only=True,
            use_detected_vpt_only=True,
        )

        assert params.gsv_api_key == "custom_key"
        assert params.confidence_detection == 0.2
        assert params.max_number_of_images == 10
        assert params.min_floor_ratio == 0.0001
        assert params.min_sky_ratio == 0.15
        assert params.device_seg == "cuda"
        assert params.device_lcnn == "cuda"
        assert params.verbose is True
        assert params.use_pitch_only is True
        assert params.use_detected_vpt_only is True

    def test_default_label_values(self):
        """Test that default label values are set correctly."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        params = HeightEstimationParameters(gsv_api_key="test_key", building_polygon=polygon)

        assert params.sky_label == [0, 2]
        assert params.building_label == [1]
        assert params.ground_label == [6, 11]

    def test_default_thresholds(self):
        """Test that default threshold values are set correctly."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        params = HeightEstimationParameters(gsv_api_key="test_key", building_polygon=polygon)

        assert params.line_classification_angle_threshold == 10.0
        assert params.line_score_threshold == 0.5
        assert params.max_dbscan_distance == 60.0
        assert params.length_threshold == 60

    def test_remapping_dict_default(self):
        """Test that the default remapping dictionary is set correctly."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        params = HeightEstimationParameters(gsv_api_key="test_key", building_polygon=polygon)

        expected_dict = {2: 1, 26: 1, 3: 2, 12: 11, 7: 11, 4: 11, 10: 11, 14: 11}
        assert params.remapping_dict == expected_dict

    def test_to_acquisition_config(self):
        """Test conversion to acquisition config."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        params = HeightEstimationParameters(
            gsv_api_key="test_key",
            building_polygon=polygon,
            min_floor_ratio=0.001,
            min_sky_ratio=0.2,
            max_number_of_images=10,
        )
        
        acq_config = params.to_acquisition_config()
        
        assert acq_config.api_key == "test_key"
        assert acq_config.min_floor_ratio == 0.001
        assert acq_config.min_sky_ratio == 0.2
        assert acq_config.max_refinement_iterations == 10

    def test_to_estimation_config(self):
        """Test conversion to estimation config."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        params = HeightEstimationParameters(
            gsv_api_key="test_key",
            building_polygon=polygon,
            device_seg="cuda",
            verbose=True,
        )
        
        est_config = params.to_estimation_config()
        
        assert est_config.device_seg == "cuda"
        assert est_config.verbose is True


class TestHeightEstimationConfig:
    """Test suite for HeightEstimationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HeightEstimationConfig()
        
        assert config.device_seg == "cpu"
        assert config.device_lcnn == "cpu"
        assert config.sky_label == [0, 2]
        assert config.building_label == [1]
        assert config.verbose is False


class TestCollectHeights:
    """Test suite for collect_heights function."""

    def test_collect_heights_single_building_single_line(self):
        """Test collecting heights from a single building with one line."""
        results = {"heights": [{"lines": [[10.5]]}]}
        heights = collect_heights(results)
        assert heights == [10.5]

    def test_collect_heights_single_building_multiple_lines(self):
        """Test collecting heights from a single building with multiple lines."""
        results = {"heights": [{"lines": [[10.5], [12.3], [9.8]]}]}
        heights = collect_heights(results)
        assert heights == [10.5, 12.3, 9.8]

    def test_collect_heights_multiple_buildings(self):
        """Test collecting heights from multiple buildings."""
        results = {"heights": [{"lines": [[10.5], [12.3]]}, {"lines": [[15.0], [14.5], [16.2]]}]}
        heights = collect_heights(results)
        assert heights == [10.5, 12.3, 15.0, 14.5, 16.2]

    def test_collect_heights_empty_results(self):
        """Test collecting heights from empty results."""
        results = {"heights": []}
        heights = collect_heights(results)
        assert heights == []

    def test_collect_heights_building_with_no_lines(self):
        """Test collecting heights when a building has no lines."""
        results = {"heights": [{"lines": []}, {"lines": [[10.5]]}]}
        heights = collect_heights(results)
        assert heights == [10.5]


class TestMeanNoOutliers:
    """Test suite for mean_no_outliers function."""

    def test_mean_no_outliers_no_outliers(self):
        """Test mean calculation when there are no outliers."""
        values = [10.0, 10.5, 11.0, 10.2, 10.8]
        result = mean_no_outliers(values)
        assert result == pytest.approx(10.5, abs=0.1)

    def test_mean_no_outliers_with_outliers(self):
        """Test mean calculation filtering out outliers."""
        values = [10.0, 10.5, 11.0, 10.2, 10.8, 50.0, 1.0]
        result = mean_no_outliers(values)
        # Should exclude 50.0 and 1.0 as outliers
        assert result == pytest.approx(10.5, abs=0.3)

    def test_mean_no_outliers_all_same_values(self):
        """Test mean calculation when all values are the same."""
        values = [10.0, 10.0, 10.0, 10.0]
        result = mean_no_outliers(values)
        assert result == pytest.approx(10.0)

    def test_mean_no_outliers_single_value(self):
        """Test mean calculation with a single value."""
        values = [10.0]
        result = mean_no_outliers(values)
        assert result == pytest.approx(10.0)

    def test_mean_no_outliers_with_extreme_outlier(self):
        """Test that extreme outliers are properly filtered."""
        values = [9.5, 10.0, 10.5, 11.0, 10.2, 100.0]
        result = mean_no_outliers(values)
        # 100.0 should be filtered out
        assert 9.0 < result < 11.5

    def test_mean_no_outliers_iqr_calculation(self):
        """Test that IQR-based filtering works correctly."""
        # Q1=10, Q3=20, IQR=10, lower=-5, upper=35
        values = [10.0, 15.0, 20.0, -10.0, 40.0]
        result = mean_no_outliers(values)
        # Should exclude -10.0 and 40.0
        assert result == pytest.approx(15.0)


class TestBuildingHeightFromSingleView:
    """Test suite for building_height_from_single_view function."""

    @pytest.fixture
    def sample_polygon(self):
        """Create a sample building polygon."""
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    @pytest.fixture
    def sample_params(self, sample_polygon):
        """Create sample HeightEstimationParameters."""
        return HeightEstimationParameters(gsv_api_key="test_key", building_polygon=sample_polygon)

    @pytest.fixture
    def mock_acquisition_result(self):
        """Create a mock acquisition result."""
        from imageable._images.acquisition import ImageAcquisitionResult
        
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_camera = CameraParameters(
            longitude=-71.05,
            latitude=42.36,
            fov=90,
            heading=45,
            pitch=10,
            width=640,
            height=480,
        )
        
        return ImageAcquisitionResult(
            image=mock_image,
            camera_params=mock_camera,
            metadata={},
            success=True,
        )

    @patch("imageable._features.height.building_height.acquire_building_image")
    @patch("imageable._features.height.building_height.estimate_height_from_image")
    def test_building_height_from_single_view_success(
        self, mock_estimate, mock_acquire, sample_params, mock_acquisition_result
    ):
        """Test successful height estimation from single view."""
        mock_acquire.return_value = mock_acquisition_result
        mock_estimate.return_value = 10.5
        
        height = building_height_from_single_view(sample_params)
        
        assert height == 10.5
        mock_acquire.assert_called_once()
        mock_estimate.assert_called_once()

    @patch("imageable._features.height.building_height.acquire_building_image")
    def test_returns_none_when_acquisition_fails(self, mock_acquire, sample_params):
        """Test that None is returned when image acquisition fails."""
        from imageable._images.acquisition import ImageAcquisitionResult
        
        mock_acquire.return_value = ImageAcquisitionResult(
            image=None,
            camera_params=CameraParameters(longitude=0, latitude=0),
            success=False,
        )
        
        result = building_height_from_single_view(sample_params)
        
        assert result is None

    @patch("imageable._features.height.building_height.acquire_building_image")
    @patch("imageable._features.height.building_height.estimate_height_from_image")
    def test_passes_correct_config_to_acquisition(
        self, mock_estimate, mock_acquire, sample_polygon, mock_acquisition_result
    ):
        """Test that acquisition receives correct configuration."""
        mock_acquire.return_value = mock_acquisition_result
        mock_estimate.return_value = 10.0
        
        params = HeightEstimationParameters(
            gsv_api_key="my_key",
            building_polygon=sample_polygon,
            min_floor_ratio=0.001,
            min_sky_ratio=0.2,
        )
        
        building_height_from_single_view(params)
        
        # Check acquisition was called with correct config
        call_kwargs = mock_acquire.call_args[1]
        config = call_kwargs["config"]
        assert config.api_key == "my_key"
        assert config.min_floor_ratio == 0.001
        assert config.min_sky_ratio == 0.2


class TestEstimateHeightFromImage:
    """Test suite for estimate_height_from_image function."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

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
            height=480,
        )

    @pytest.fixture
    def sample_polygon(self):
        """Create a sample polygon."""
        return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    @pytest.fixture
    def mock_models(self):
        """Create mocks for all ML models used in estimation."""
        with (
            patch("imageable._features.height.building_height.SegformerSegmentationWrapper") as mock_segformer,
            patch("imageable._features.height.building_height.LCNNWrapper") as mock_lcnn,
            patch("imageable._features.height.building_height.VPTSWrapper") as mock_vpts,
            patch("imageable._features.height.building_height.HeightCalculator") as mock_calculator,
            patch("imageable._features.height.building_height._predict_line_score_threshold") as mock_line_thresh,
        ):
            # Mock segmentation
            mock_seg_instance = Mock()
            mock_segformer.return_value = mock_seg_instance
            mock_seg_instance.predict.return_value = np.zeros((480, 640), dtype=np.uint8)
            mock_seg_instance._remap_labels.return_value = np.zeros((480, 640), dtype=np.uint8)
            
            # Mock LCNN
            mock_lcnn_instance = Mock()
            mock_lcnn.return_value = mock_lcnn_instance
            mock_lcnn_instance.predict.return_value = {
                "processed_lines": np.array([[0, 0, 100, 100]]),
                "processed_scores": np.array([0.95]),
            }
            
            # Mock VPTS
            mock_vpts_instance = Mock()
            mock_vpts.return_value = mock_vpts_instance
            mock_vpts_instance.predict.return_value = {
                "vpts_2d": np.array([[320, 240], [320, 0], [0, 240]])
            }
            
            # Mock height calculator
            mock_calc_instance = Mock()
            mock_calculator.return_value = mock_calc_instance
            mock_calc_instance.calculate_heights.return_value = {
                "heights": [{"lines": [[10.5], [11.2], [10.8]]}]
            }
            
            # Mock line score threshold
            mock_line_thresh.return_value = 0.5
            
            yield {
                "segformer": mock_seg_instance,
                "lcnn": mock_lcnn_instance,
                "vpts": mock_vpts_instance,
                "calculator": mock_calc_instance,
            }

    def test_estimate_height_success(
        self, sample_image, sample_camera_params, sample_polygon, mock_models
    ):
        """Test successful height estimation."""
        height = estimate_height_from_image(
            sample_image,
            sample_camera_params,
            sample_polygon,
        )
        
        assert isinstance(height, float)
        assert height > 0

    def test_estimate_height_with_custom_config(
        self, sample_image, sample_camera_params, sample_polygon, mock_models
    ):
        """Test estimation with custom configuration."""
        config = HeightEstimationConfig(
            device_seg="cuda",
            verbose=True,
        )
        
        height = estimate_height_from_image(
            sample_image,
            sample_camera_params,
            sample_polygon,
            config=config,
        )
        
        # Verify verbose was passed
        call_kwargs = mock_models["calculator"].calculate_heights.call_args[1]
        assert call_kwargs["verbose"] is True

    def test_returns_none_for_none_image(
        self, sample_camera_params, sample_polygon
    ):
        """Test that None is returned for None image."""
        result = estimate_height_from_image(
            None,
            sample_camera_params,
            sample_polygon,
        )
        
        assert result is None

    def test_returns_none_when_calculator_fails(
        self, sample_image, sample_camera_params, sample_polygon, mock_models
    ):
        """Test that None is returned when calculator returns None."""
        mock_models["calculator"].calculate_heights.return_value = None
        
        result = estimate_height_from_image(
            sample_image,
            sample_camera_params,
            sample_polygon,
        )
        
        assert result is None


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    @pytest.fixture
    def realistic_polygon(self):
        """Create a realistic building polygon."""
        return Polygon([(42.3601, -71.0589), (42.3601, -71.0580), (42.3610, -71.0580), (42.3610, -71.0589)])

    def test_with_realistic_parameters(self, realistic_polygon):
        """Test with realistic building parameters."""
        params = HeightEstimationParameters(
            gsv_api_key="test_key",
            building_polygon=realistic_polygon,
            device_seg="cpu",
            device_lcnn="cpu",
            verbose=False,
        )

        # Verify all parameters are set correctly
        assert params.gsv_api_key == "test_key"
        assert params.building_polygon == realistic_polygon
        assert params.device_seg == "cpu"
        assert params.device_lcnn == "cpu"
