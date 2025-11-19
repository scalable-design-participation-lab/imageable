from unittest.mock import Mock, patch

import numpy as np
import pytest
from shapely import Polygon

from imageable._features.height.building_height import (
    HeightEstimationParameters,
    building_height_from_single_view,
    collect_heights,
    mean_no_outliers,
)


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
        assert params.line_score_threshold == 0.94
        assert params.max_dbscan_distance == 60.0
        assert params.length_threshold == 60

    def test_remapping_dict_default(self):
        """Test that the default remapping dictionary is set correctly."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        params = HeightEstimationParameters(gsv_api_key="test_key", building_polygon=polygon)

        expected_dict = {2: 1, 26: 1, 3: 2, 12: 11, 7: 11, 4: 11, 10: 11, 14: 11}
        assert params.remapping_dict == expected_dict


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
    def mock_components(self):
        """Create mocks for all components used in height estimation."""
        with (
            patch("imageable.features.height.building_height.CameraParametersRefiner") as mock_refiner,
            patch("imageable.features.height.building_height.SegformerSegmentationWrapper") as mock_segformer,
            patch("imageable.features.height.building_height.LCNNWrapper") as mock_lcnn,
            patch("imageable.features.height.building_height.VPTSWrapper") as mock_vpts,
            patch("imageable.features.height.building_height.HeightCalculator") as mock_calculator,
        ):
            # Mock camera parameter refiner
            mock_refiner_instance = Mock()
            mock_refiner.return_value = mock_refiner_instance

            mock_camera_params = Mock()
            mock_camera_params.fov = 90
            mock_camera_params.pitch = 0
            mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            mock_refiner_instance.adjust_parameters.return_value = (mock_camera_params, True, mock_image)

            # Mock segmentation model
            mock_segformer_instance = Mock()
            mock_segformer.return_value = mock_segformer_instance
            mock_segformer_instance.predict.return_value = np.zeros((480, 640), dtype=np.uint8)
            mock_segformer_instance._remap_labels.return_value = np.zeros((480, 640), dtype=np.uint8)

            # Mock LCNN model
            mock_lcnn_instance = Mock()
            mock_lcnn.return_value = mock_lcnn_instance
            mock_lcnn_instance.predict.return_value = {
                "processed_lines": np.array([[0, 0, 100, 100]]),
                "processed_scores": np.array([0.95]),
            }

            # Mock VPTS model
            mock_vpts_instance = Mock()
            mock_vpts.return_value = mock_vpts_instance
            mock_vpts_instance.predict.return_value = {"vpts_2d": np.array([[320, 240], [320, 0], [0, 240]])}

            # Mock height calculator
            mock_calculator_instance = Mock()
            mock_calculator.return_value = mock_calculator_instance
            mock_calculator_instance.calculate_heights.return_value = {"heights": [{"lines": [[10.5], [11.2], [10.8]]}]}

            yield {
                "refiner": mock_refiner_instance,
                "segformer": mock_segformer_instance,
                "lcnn": mock_lcnn_instance,
                "vpts": mock_vpts_instance,
                "calculator": mock_calculator_instance,
                "image": mock_image,
            }

    def test_building_height_from_single_view_success(self, sample_params, mock_components):
        """Test successful height estimation from single view."""
        height = building_height_from_single_view(sample_params)

        # Should return a reasonable height value
        assert isinstance(height, (float, np.floating))
        assert height > 0
        assert height == pytest.approx(10.83, abs=0.5)

    def test_camera_parameter_refinement_called(self, sample_params, mock_components):
        """Test that camera parameter refinement is called with correct parameters."""
        building_height_from_single_view(sample_params)

        mock_components["refiner"].adjust_parameters.assert_called_once()
        call_kwargs = mock_components["refiner"].adjust_parameters.call_args[1]

        assert call_kwargs["confidence_detection"] == 0.1
        assert call_kwargs["max_number_of_images"] == 5

    def test_segmentation_model_loaded_and_used(self, sample_params, mock_components):
        """Test that segmentation model is properly loaded and used."""
        building_height_from_single_view(sample_params)

        mock_components["segformer"].load_model.assert_called_once()
        mock_components["segformer"].predict.assert_called_once()
        mock_components["segformer"]._remap_labels.assert_called_once()

    def test_lcnn_model_loaded_and_used(self, sample_params, mock_components):
        """Test that LCNN model is properly loaded and used."""
        building_height_from_single_view(sample_params)

        mock_components["lcnn"].load_model.assert_called_once()
        mock_components["lcnn"].predict.assert_called_once()

    def test_vpts_model_used_with_correct_fov(self, sample_params, mock_components):
        """Test that VPTS model receives correct FOV parameter."""
        building_height_from_single_view(sample_params)

        call_kwargs = mock_components["vpts"].predict.call_args[1]
        assert call_kwargs["FOV"] == 90
        assert call_kwargs["seed"] == 42
        assert call_kwargs["length_threshold"] == 60

    def test_height_calculator_receives_correct_inputs(self, sample_params, mock_components):
        """Test that height calculator receives all required inputs."""
        building_height_from_single_view(sample_params)

        mock_components["calculator"].calculate_heights.assert_called_once()
        call_kwargs = mock_components["calculator"].calculate_heights.call_args[1]

        assert "data" in call_kwargs
        assert "camera" in call_kwargs
        assert call_kwargs["verbose"] is False
        assert "pitch" in call_kwargs

    def test_custom_parameters_applied(self, sample_polygon, mock_components):
        """Test that custom parameters are correctly applied."""
        custom_params = HeightEstimationParameters(
            gsv_api_key="custom_key",
            building_polygon=sample_polygon,
            verbose=True,
            use_pitch_only=True,
            use_detected_vpt_only=True,
            max_number_of_images=10,
        )

        building_height_from_single_view(custom_params)

        # Check verbose parameter
        call_kwargs = mock_components["calculator"].calculate_heights.call_args[1]
        assert call_kwargs["verbose"] is True
        assert call_kwargs["use_pitch_only"] is True
        assert call_kwargs["use_detected_vpt_only"] is True

    def test_focal_length_calculation(self, sample_params, mock_components):
        """Test that focal length is calculated correctly."""
        building_height_from_single_view(sample_params)

        call_kwargs = mock_components["calculator"].calculate_heights.call_args[1]
        camera_params = call_kwargs["camera"]

        # Focal length should be positive and reasonable
        assert camera_params.focal_length > 0
        assert camera_params.focal_length < 5000

    def test_focal_length_fallback(self, sample_params, mock_components):
        """Test that focal length falls back to 90 if out of range."""
        # Create a mock that would result in invalid focal length
        mock_components["refiner"].adjust_parameters.return_value = (
            Mock(fov=0.01, pitch=0),  # Very small FOV
            True,
            mock_components["image"],
        )

        building_height_from_single_view(sample_params)

        call_kwargs = mock_components["calculator"].calculate_heights.call_args[1]
        camera_params = call_kwargs["camera"]

        # Should fall back to 90
        assert camera_params.focal_length == 90

    def test_segmentation_label_shifting(self, sample_params, mock_components):
        """Test that segmentation labels are shifted when enabled."""
        sample_params.shift_segmentation_labels = True

        # Mock segmentation to return specific values
        mock_seg_result = np.array([[0, 1, 2]], dtype=np.uint8)
        mock_components["segformer"].predict.return_value = mock_seg_result

        building_height_from_single_view(sample_params)

        # After shifting, values should be 1, 2, 3 before remapping
        # We can't directly check this, but we can verify remap was called
        mock_components["segformer"]._remap_labels.assert_called_once()

    def test_min_floor_sky_ratio_applied(self, sample_params, mock_components):
        """Test that min floor and sky ratios are applied to refiner."""
        sample_params.min_floor_ratio = 0.001
        sample_params.min_sky_ratio = 0.2

        building_height_from_single_view(sample_params)

        # Check that the ratios were set on the refiner
        assert mock_components["refiner"].MIN_FLOOR_RATIO == 0.001
        assert mock_components["refiner"].MIN_SKY_RATIO == 0.2

    def test_multiple_buildings_height_calculation(self, sample_params, mock_components):
        """Test height calculation with multiple buildings."""
        # Mock calculator to return multiple buildings
        mock_components["calculator"].calculate_heights.return_value = {
            "heights": [{"lines": [[10.0], [10.5]]}, {"lines": [[15.0], [14.5]]}]
        }

        height = building_height_from_single_view(sample_params)

        # Should average all heights
        assert isinstance(height, (float, np.floating))
        assert height > 0


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
