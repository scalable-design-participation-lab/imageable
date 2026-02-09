"""
Building height estimation from street view images.

This module provides functions for estimating building heights using
single-view metrology techniques applied to street-level imagery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import json
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from shapely import Polygon

from imageable._images.acquisition import (
    ImageAcquisitionConfig,
    ImageAcquisitionResult,
    acquire_building_image,
    load_image_with_metadata,
)
from imageable._images.camera.camera_parameters import CameraParameters as GSVCameraParameters
from imageable._models.height_estimation.height_calculator import (
    CameraParameters,
    HeightCalculator,
    HeightEstimationInput,
)
from imageable._models.huggingface.segformer_segmentation import SegformerSegmentationWrapper
from imageable._models.lcnn.lcnn_wrapper import LCNNWrapper
from imageable._models.vpts.vpts_wrapper import VPTSWrapper
from imageable._extraction.extract import extract_building_properties


# Constants
MIN_FOCAL_LENGTH = 1
MAX_FOCAL_LENGTH = 5000
DEFAULT_FOCAL_LENGTH = 90
DEFAULT_LINE_SCORE_THRESHOLD = 0.1


@dataclass
class HeightEstimationConfig:
    """
    Configuration for the height estimation algorithm.

    These parameters control the segmentation, line detection, and
    height calculation steps - not image acquisition.

    Parameters
    ----------
    segformer_name
        Name of the SegFormer model for building segmentation.
    palette_path
        Path to the palette file for visualization.
    device_seg
        Device for segmentation model ('cpu' or 'cuda').
    remapping_dict
        Dictionary to remap segmentation labels.
    lcnn_config_path
        Path to LCNN configuration file.
    device_lcnn
        Device for LCNN model ('cpu' or 'cuda').
    lcnn_checkpoint_path
        Path to LCNN checkpoint.
    length_threshold
        Minimum line length for vanishing point estimation.
    seed_vp_ransac
        Random seed for RANSAC.
    sky_label
        Segmentation labels representing sky.
    building_label
        Segmentation labels representing buildings.
    ground_label
        Segmentation labels representing ground.
    line_classification_angle_threshold
        Angle threshold for classifying vertical lines.
    line_score_threshold
        Score threshold for filtering lines.
    edge_threshold
        Edge threshold for line refinement.
    max_dbscan_distance
        Maximum distance for DBSCAN clustering.
    shift_segmentation_labels
        Whether to shift segmentation labels by 1.
    verbose
        Enable verbose output.
    use_pitch_only
        Use only pitch for height estimation.
    use_detected_vpt_only
        Use only detected vanishing point.
    """

    segformer_name: str = "nvidia/segformer-b5-finetuned-ade-640-640"
    palette_path: str = str(Path(__file__).resolve().parents[2] / "assets" / "ade20k_palette.json")
    device_seg: str = "cpu"
    remapping_dict: dict[int, int] = field(
        default_factory=lambda: {2: 1, 26: 1, 3: 2, 12: 11, 7: 11, 4: 11, 10: 11, 14: 11}
    )
    lcnn_config_path: str = str(Path(__file__).resolve().parents[2] / "assets" / "wireframe.yaml")
    device_lcnn: str = "cpu"
    lcnn_checkpoint_path: str = ""
    length_threshold: int = 60
    seed_vp_ransac: int = 42
    sky_label: list[int] = field(default_factory=lambda: [0, 2])
    building_label: list[int] = field(default_factory=lambda: [1])
    ground_label: list[int] = field(default_factory=lambda: [6, 11])
    line_classification_angle_threshold: float = 10.0
    line_score_threshold: float = 0.5
    edge_threshold: str = "2,2"
    max_dbscan_distance: float = 60.0
    shift_segmentation_labels: bool = True
    verbose: bool = False
    use_pitch_only: bool = False
    use_detected_vpt_only: bool = False


@dataclass
class HeightEstimationParameters:
    """
    Combined parameters for end-to-end height estimation.

    This dataclass combines image acquisition and height estimation
    parameters for the convenience function `building_height_from_single_view`.

    For more control, use `acquire_building_image` and `estimate_height_from_image`
    separately.
    """

    gsv_api_key: str
    building_polygon: Polygon
    pictures_directory: str = str(Path(__file__).resolve().parents[3] / "notebooks" / "pictures")
    # Image acquisition parameters
    save_reel: bool = False
    overwrite_images: bool = True
    confidence_detection: float = 0.1
    max_number_of_images: int = 5
    polygon_buffer_constant: float = 20
    min_floor_ratio: float = 0.00001
    min_sky_ratio: float = 0.1
    # Height estimation parameters
    segformer_name: str = "nvidia/segformer-b5-finetuned-ade-640-640"
    palette_path: str = str(Path(__file__).resolve().parents[2] / "assets" / "ade20k_palette.json")
    device_seg: str = "cpu"
    remapping_dict: dict[int, int] = field(
        default_factory=lambda: {2: 1, 26: 1, 3: 2, 12: 11, 7: 11, 4: 11, 10: 11, 14: 11}
    )
    lcnn_config_path: str = str(Path(__file__).resolve().parents[2] / "assets" / "wireframe.yaml")
    device_lcnn: str = "cpu"
    lcnn_checkpoint_path: str = ""
    length_threshold: int = 60
    seed_vp_ransac: int = 42
    sky_label: list[int] = field(default_factory=lambda: [0, 2])
    building_label: list[int] = field(default_factory=lambda: [1])
    ground_label: list[int] = field(default_factory=lambda: [6, 11])
    line_classification_angle_threshold: float = 10.0
    line_score_threshold: float = 0.5
    edge_threshold: str = "2,2"
    max_dbscan_distance: float = 60.0
    verbose: bool = False
    use_pitch_only: bool = False
    use_detected_vpt_only: bool = False
    shift_segmentation_labels: bool = True
    all_buildings: list[Polygon] = None

    def to_acquisition_config(self) -> ImageAcquisitionConfig:
        """Convert to ImageAcquisitionConfig."""
        return ImageAcquisitionConfig(
            api_key=self.gsv_api_key,
            save_directory=self.pictures_directory,
            save_intermediate=self.save_reel,
            overwrite=self.overwrite_images,
            min_floor_ratio=self.min_floor_ratio,
            min_sky_ratio=self.min_sky_ratio,
            max_refinement_iterations=self.max_number_of_images,
            confidence_threshold=self.confidence_detection,
            polygon_buffer_constant=self.polygon_buffer_constant,
        )

    def to_estimation_config(self) -> HeightEstimationConfig:
        """Convert to HeightEstimationConfig."""
        return HeightEstimationConfig(
            segformer_name=self.segformer_name,
            palette_path=self.palette_path,
            device_seg=self.device_seg,
            remapping_dict=self.remapping_dict,
            lcnn_config_path=self.lcnn_config_path,
            device_lcnn=self.device_lcnn,
            lcnn_checkpoint_path=self.lcnn_checkpoint_path,
            length_threshold=self.length_threshold,
            seed_vp_ransac=self.seed_vp_ransac,
            sky_label=self.sky_label,
            building_label=self.building_label,
            ground_label=self.ground_label,
            line_classification_angle_threshold=self.line_classification_angle_threshold,
            line_score_threshold=self.line_score_threshold,
            edge_threshold=self.edge_threshold,
            max_dbscan_distance=self.max_dbscan_distance,
            shift_segmentation_labels=self.shift_segmentation_labels,
            verbose=self.verbose,
            use_pitch_only=self.use_pitch_only,
            use_detected_vpt_only=self.use_detected_vpt_only,
        )


def estimate_height_from_image(
    image: NDArray[np.uint8],
    camera_params: GSVCameraParameters,
    polygon: Polygon,
    config: HeightEstimationConfig | None = None,
    all_buildings: list[Polygon] | None = None,
) -> float | None:
    """
    Estimate building height from a street view image.

    This is the core height estimation function that operates on a pre-acquired
    image. Use this when you have your own image or want to separate image
    acquisition from analysis.

    Parameters
    ----------
    image
        Street view image as numpy array (H, W, 3) in RGB format.
    camera_params
        Camera parameters used to capture the image.
    polygon
        Building footprint polygon.
    config
        Height estimation configuration. Uses defaults if None.
    all_buildings
        List of all building polygons for contextual features.
        Used for line score threshold prediction.

    Returns
    -------
    height
        Estimated building height in meters, or None if estimation failed.

    Examples
    --------
    >>> from imageable._images.acquisition import acquire_building_image, ImageAcquisitionConfig
    >>> from imageable._features.height.building_height import estimate_height_from_image
    >>>
    >>> # Acquire image
    >>> acq_config = ImageAcquisitionConfig(api_key="your_key")
    >>> result = acquire_building_image(polygon, acq_config)
    >>>
    >>> # Estimate height
    >>> if result.is_valid:
    ...     height = estimate_height_from_image(
    ...         result.image,
    ...         result.camera_params,
    ...         polygon
    ...     )
    """
    from imageable._models.line_param_selection_model import LineParameterSelectionModel

    if config is None:
        config = HeightEstimationConfig()

    if image is None:
        return None

    # Load and run segmentation
    segmentation_model = SegformerSegmentationWrapper(
        model_name=config.segformer_name,
        device=config.device_seg,
        palette_path=config.palette_path,
    )
    segmentation_model.load_model()

    seg_results = segmentation_model.predict(image)
    if config.shift_segmentation_labels:
        seg_results = seg_results.squeeze().astype("uint8") + 1
    remapped_seg = segmentation_model._remap_labels(seg_results, config.remapping_dict)

    # Line detection with LCNN
    lcnn_model = LCNNWrapper(
        config_path=config.lcnn_config_path,
        device=config.device_lcnn,
        checkpoint_path=config.lcnn_checkpoint_path,
    )
    lcnn_model.load_model()
    lcnn_results = lcnn_model.predict(image)

    # Vanishing point detection
    vpts_model = VPTSWrapper()
    vpts_dictionary = vpts_model.predict(
        image,
        FOV=camera_params.fov,
        seed=config.seed_vp_ransac,
        length_threshold=config.length_threshold,
    )

    # Predict optimal line score threshold using building features
    line_score_threshold = _predict_line_score_threshold(
        polygon, all_buildings, config.building_label
    )

    # Build height calculator configuration
    calc_config = {
        "STREET_VIEW": {"HVFoV": str(camera_params.fov), "CameraHeight": "2.5"},
        "SEGMENTATION": {
            "SkyLabel": ",".join(map(str, config.sky_label)),
            "BuildingLabel": ",".join(map(str, config.building_label)),
            "GroundLabel": ",".join(map(str, config.ground_label)),
        },
        "LINE_CLASSIFY": {
            "AngleThres": str(config.line_classification_angle_threshold),
            "LineScore": str(line_score_threshold),
        },
        "LINE_REFINE": {"Edge_Thres": config.edge_threshold},
        "HEIGHT_MEAS": {"MaxDBSANDist": str(config.max_dbscan_distance)},
    }

    height_calculator = HeightCalculator(calc_config)

    # Prepare input data
    height_input = HeightEstimationInput(
        image=image,
        segmentation=remapped_seg,
        vanishing_points=vpts_dictionary["vpts_2d"],
        lines=lcnn_results["processed_lines"],
        line_scores=lcnn_results["processed_scores"],
    )

    # Calculate focal length from FOV
    _h, width = image.shape[:2]
    focal_length = 0.5 * width / np.tan(0.5 * np.radians(camera_params.fov))
    if focal_length < MIN_FOCAL_LENGTH or focal_length > MAX_FOCAL_LENGTH:
        focal_length = DEFAULT_FOCAL_LENGTH

    calc_camera = CameraParameters(
        focal_length=focal_length,
        cx=image.shape[1] / 2,
        cy=image.shape[0] / 2,
    )

    # Run height calculation
    results = height_calculator.calculate_heights(
        data=height_input,
        camera=calc_camera,
        verbose=config.verbose,
        pitch=camera_params.pitch,
        use_pitch_only=config.use_pitch_only,
        use_detected_vpt_only=config.use_detected_vpt_only,
    )

    if results is None:
        return None

    heights = collect_heights(results)
    if len(heights) == 0:
        return None

    return mean_no_outliers(heights)


def _predict_line_score_threshold(
    polygon: Polygon,
    all_buildings: list[Polygon] | None,
    building_label: list[int],
) -> float:
    """Predict optimal line score threshold based on building features."""
    from imageable._models.line_param_selection_model import LineParameterSelectionModel

    try:
        line_model = LineParameterSelectionModel()
        line_model.load_model()

        props = extract_building_properties(
            building_id=building_label,
            polygon=polygon,
            all_buildings=all_buildings,
            height_value=None,
            verbose=False,
        )

        props_dict = props.to_dict()
        features = line_model.corr_model.FEATURES_USED
        vector = np.array([[props_dict[f] for f in features]])
        return line_model.predict(vector.reshape(1, -1))

    except Exception:
        return DEFAULT_LINE_SCORE_THRESHOLD


def building_height_from_single_view(
    height_estimation_params: HeightEstimationParameters,
) -> float | None:
    """
    Estimate building height from a single street-level image.

    This is a convenience function that combines image acquisition and
    height estimation. For more control, use `acquire_building_image`
    and `estimate_height_from_image` separately.

    Parameters
    ----------
    height_estimation_params
        Parameters for both image acquisition and height estimation.

    Returns
    -------
    height
        Estimated building height in meters, or None if estimation failed.
    """
    # Acquire image (handles caching internally)
    acq_config = height_estimation_params.to_acquisition_config()
    acq_result = acquire_building_image(
        polygon=height_estimation_params.building_polygon,
        config=acq_config,
    )

    if not acq_result.is_valid:
        return None

    # Run height estimation
    est_config = height_estimation_params.to_estimation_config()
    return estimate_height_from_image(
        image=acq_result.image,
        camera_params=acq_result.camera_params,
        polygon=height_estimation_params.building_polygon,
        config=est_config,
        all_buildings=height_estimation_params.all_buildings,
    )


def corrected_height_from_single_view(
    height_estimation_parameters: HeightEstimationParameters,
    building_id: int,
    all_buildings: list[Polygon],
    crs: str = "EPSG:4326",
    verbose: bool = False,
    correction_model: "HeightCorrectionModel" = None,
) -> float:
    """
    Compute a corrected building height from a single Street View image.

    This function:
    1. Computes the raw height using `building_height_from_single_view`.
    2. Loads the image and metadata from cache.
    3. Computes material percentages from the image.
    4. Applies a correction model to refine the height estimate.
    """
    from imageable._models.height_correction_model import HeightCorrectionModel
    from imageable._features.materials.building_materials import (
        BuildingMaterialProperties,
        get_building_materials_segmentation,
    )

    if correction_model is None:
        correction_model = HeightCorrectionModel()
    correction_model.load_model()
    correction_model.pretrained.feature_indices_used_for_clustering = list(range(0, 13))

    # Get raw height estimate
    raw_height = building_height_from_single_view(height_estimation_parameters)

    # Try to load image and compute material percentages
    street_view_image = None
    material_percentages = None

    pictures_dir = Path(height_estimation_parameters.pictures_directory)

    try:
        acq_result = load_image_with_metadata(
            pictures_dir / "image.jpg",
            pictures_dir / "metadata.json",
        )

        if acq_result.is_valid:
            street_view_image = acq_result.image

            # Extract camera parameters for materials
            cam_dict = acq_result.metadata.get("camera_parameters", {})
            camera_parameters = CameraParameters(
                longitude=cam_dict.get("longitude", 0),
                latitude=cam_dict.get("latitude", 0),
            )
            camera_parameters.fov = cam_dict.get("fov", 90)
            camera_parameters.heading = cam_dict.get("heading", 0)
            camera_parameters.pitch = cam_dict.get("pitch", 0)
            camera_parameters.height = cam_dict.get("height", 640)
            camera_parameters.width = cam_dict.get("width", 640)

            bmp = BuildingMaterialProperties(
                img=street_view_image,
                verbose=verbose,
            )
            bmp.building_height = raw_height
            bmp.footprint = height_estimation_parameters.building_polygon
            bmp.camera_parameters = camera_parameters

            material_percentages = get_building_materials_segmentation(bmp)

    except Exception:
        #print("WAS NONE")
        pictures_dir = Path(height_estimation_parameters.pictures_directory)
        try:
            image_path = pictures_dir / "image.jpg"
            metadata_path = pictures_dir / "metadata.json"

            street_view_image = np.array(Image.open(image_path))

            with open(metadata_path) as f:
                metadata = json.load(f)

            camera_parameters_dict = metadata["camera_parameters"]
            camera_parameters = GSVCameraParameters(
                longitude=camera_parameters_dict["longitude"],
                latitude=camera_parameters_dict["latitude"],
            )
            camera_parameters.fov = camera_parameters_dict["fov"]
            camera_parameters.heading = camera_parameters_dict["heading"]
            camera_parameters.pitch = camera_parameters_dict["pitch"]
            camera_parameters.height = camera_parameters_dict["height"]
            camera_parameters.width = camera_parameters_dict["width"]

            bmp = BuildingMaterialProperties(
                img=street_view_image,
                verbose=verbose,
            )
            bmp.building_height = raw_height
            bmp.footprint = height_estimation_parameters.building_polygon
            bmp.camera_parameters = None

            material_percentages = get_building_materials_segmentation(bmp)

        except Exception:
            street_view_image = None
            material_percentages = None

    # Apply correction model
    corrected_height = correction_model.predict(
        raw_height=raw_height,
        estimation_params=height_estimation_parameters,
        building_id=building_id,
        all_buildings=all_buildings,
        crs=crs,
        street_view_image=street_view_image,
        material_percentages=material_percentages,
        verbose=verbose,
    )

    return corrected_height


# ============================================================================
# Utility functions
# ============================================================================


def collect_heights(results: dict) -> list[float]:
    """
    Collect all estimated heights from the results dictionary.

    Parameters
    ----------
    results
        Dictionary containing height estimation results.

    Returns
    -------
    heights
        List of all estimated heights.
    """
    return [line[0] for building in results["heights"] for line in building["lines"]]


def get_filtered_lines(results: dict):
    """
    Return the final filtered line segments used by the height calculator.

    Each element is (a, b) where a and b are endpoints in image coords.
    """
    lines = []
    for building in results["heights"]:
        for line in building["lines"]:
            _ht, a, b, *_ = line
            lines.append((a, b))
    return lines


def count_filtered_lines(results: dict) -> int:
    """
    Number of filtered lines N_l used in the final height computation.
    """
    return sum(len(building["lines"]) for building in results["heights"])


def mean_no_outliers(values: list[float] | NDArray[np.floating]) -> float:
    """
    Compute the mean of a list of values after removing outliers using
    the interquartile range (IQR) method.

    Parameters
    ----------
    values
        List of numerical values.

    Returns
    -------
    mean_value
        Mean of the values after outlier removal.
    """
    values = np.array(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = values[(values >= lower) & (values <= upper)]
    return float(np.mean(filtered))


def estimate_height_and_lines_from_image(
    image_path: Union[str, Path],
    segformer_model: SegformerSegmentationWrapper,
    lcnn_model: LCNNWrapper,
    vpts_model: VPTSWrapper,
    remapping_dict: dict[int, int],
    sky_label: list[int],
    building_label: list[int],
    ground_label: list[int],
    line_classification_angle_threshold: float,
    line_score_threshold: float,
    edge_threshold: str,
    max_dbscan_distance: float,
    fov: float = 90.0,
    verbose: bool = False,
) -> tuple[float | None, int]:
    """
    Estimate height and count lines from an image file.

    This is a lower-level function that accepts pre-loaded models.
    """
    image_path = Path(image_path)
    image: NDArray[np.uint8] = np.array(Image.open(image_path).convert("RGB"))

    # Segmentation
    seg_raw: NDArray[np.int_] = segformer_model.predict(image)
    seg_raw = seg_raw.squeeze().astype("uint8") + 1
    seg: NDArray[np.int_] = segformer_model._remap_labels(seg_raw, remapping_dict)

    # LCNN
    lcnn_results: dict = lcnn_model.predict(image)
    lines: NDArray[np.floating] = lcnn_results["processed_lines"]
    line_scores: NDArray[np.floating] = lcnn_results["processed_scores"]

    # Vanishing points
    vpts_dictionary: dict = vpts_model.predict(
        image,
        FOV=fov,
        seed=42,
        length_threshold=60,
    )
    vps_2d: NDArray[np.floating] = vpts_dictionary["vpts_2d"]

    # Camera
    h, w = image.shape[:2]
    camera: CameraParameters = CameraParameters.from_fov(
        fov_degrees=fov,
        image_width=w,
        image_height=h,
    )

    # Config for height calculator
    config_calculation: dict[str, dict[str, str]] = {
        "STREET_VIEW": {"HVFoV": str(fov), "CameraHeight": "2.5"},
        "SEGMENTATION": {
            "SkyLabel": ",".join(map(str, sky_label)),
            "BuildingLabel": ",".join(map(str, building_label)),
            "GroundLabel": ",".join(map(str, ground_label)),
        },
        "LINE_CLASSIFY": {
            "AngleThres": str(line_classification_angle_threshold),
            "LineScore": str(line_score_threshold),
        },
        "LINE_REFINE": {"Edge_Thres": edge_threshold},
        "HEIGHT_MEAS": {"MaxDBSANDist": str(max_dbscan_distance)},
    }

    height_calculator = HeightCalculator(config_calculation)

    data = HeightEstimationInput(
        image=image,
        segmentation=seg,
        vanishing_points=vps_2d,
        lines=lines,
        line_scores=line_scores,
    )

    results = height_calculator.calculate_heights(
        data=data,
        camera=camera,
        verbose=verbose,
        pitch=25.0,
        use_pitch_only=False,
        use_detected_vpt_only=False,
    )

    if results is None:
        return None, 0

    heights = collect_heights(results)
    if len(heights) == 0:
        return None, 0

    height_estimate: float = mean_no_outliers(heights)
    n_lines: int = count_filtered_lines(results)
    return height_estimate, n_lines
