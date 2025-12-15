from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from shapely import Polygon
from imageable._images.camera.camera_adjustment import CameraParametersRefiner
from typing import Union
from imageable._models.height_estimation.height_calculator import (
    CameraParameters,
    HeightCalculator,
    HeightEstimationInput,
)
from imageable._images.camera.camera_parameters import CameraParameters as GSVCameraParameters

from imageable._models.huggingface.segformer_segmentation import SegformerSegmentationWrapper
from imageable._models.lcnn.lcnn_wrapper import LCNNWrapper
from imageable._models.vpts.vpts_wrapper import VPTSWrapper
from PIL import Image
import json
from imageable._features.materials.building_materials import BuildingMaterialProperties, get_building_materials_segmentation
from imageable._extraction.extract import extract_building_properties


@dataclass
class HeightEstimationParameters:
    """
    Encapsulates parameters required for the height estimation process.

    Parameters
    ----------
    gsv_api_key
        Google Street View API key.
    building_polygon
        Shapely polygon representing the building footprint.
    pictures_directory
        Directory where pictures are stored.
    save_reel
        Whether to save intermediate images during parameter refinement.
    overwrite_images
        Overwrite existing images in the pictures directory.
    confidence_detection
        Confidence threshold for detections of sky and ground during the parameter refinement.
    max_number_of_images
        Maximum number of images to use as part of the parameter refinement.
    polygon_buffer_constant
        Buffer constant to apply based on the polygon area to retrieve a street network around
        the building. This is used to determine the best observation point for taking a picture
        of the building.
    min_floor_ratio
        Min ratio of floor pixels to consider that the ground is visible in the image.
    min_sky_ratio
        Min ratio of sky pixels to consider that the sky is visible in the image.
    segformer_name
        Name of the SegFormer model to use for building segmentation.
    palette_path
        Path to the palette file for visualization of segmentation results.
    remapping_dict
        Dictionary to remap segmentation labels to desired classes.
    device_seg
        Device to use for building segmentation (e.g., 'cpu' or 'cuda').
    lcnn_config_path
        Path to the LCNN configuration file.
    device_lcnn
        Device to use for LCNN line segmentation (e.g., 'cpu' or 'cuda').
    lcnn_checkpoint_path
        Path to the LCNN model checkpoint.
    length_threshold
        Minimum length of line segments to consider for vanishing point estimation.
    seed_vp_ransac
        Random seed for RANSAC in vanishing point estimation.
    sky_label
        List of integer labels representing sky in the segmentation map.
    building_label
        List of integer labels representing buildings in the segmentation map.
    ground_label
        List of integer labels representing ground in the segmentation map.
    line_classification_angle_threshold
        Angle threshold (in degrees) for classifying lines as vertical.
    line_score_threshold
        Score threshold for filtering line segments.
    edge_thres
        Edge threshold for image borders.
    max_dbscan_distance
        Maximum distance for DBSCAN clustering of line segments.
    verbose
        Whether to enable verbose logging.
    use_pitch_only
        Whether to use only the pitch angle for height estimation.
    use_detected_vpt_only
        Whether to use only the detected vanishing point for height estimation.
    shift_segmentation_labels
        Whether to shift segmentation labels by 1 to match expected input.
    """

    gsv_api_key: str
    building_polygon: Polygon
    pictures_directory: str = str(Path(__file__).resolve().parents[3] / "notebooks" / "pictures")
    # Parameter refinement parameters
    save_reel: bool = False
    overwrite_images: bool = True
    confidence_detection: float = 0.1
    max_number_of_images: int = 5
    polygon_buffer_constant: float = 20
    min_floor_ratio: float = 0.00001
    min_sky_ratio: float = 0.1
    # Building Segmentation parameters
    segformer_name: str = "nvidia/segformer-b5-finetuned-ade-640-640"
    palette_path: str = str(Path(__file__).resolve().parents[2] / "assets" / "ade20k_palette.json")
    device_seg: str = "cpu"
    remapping_dict: dict[int, int] = field(
        default_factory=lambda: {2: 1, 26: 1, 3: 2, 12: 11, 7: 11, 4: 11, 10: 11, 14: 11}
    )
    # Line segmentation parameters
    lcnn_config_path: str = str(Path(__file__).resolve().parents[2] / "assets" / "wireframe.yaml")
    device_lcnn: str = "cpu"
    lcnn_checkpoint_path: str = ""
    # Vanishing point parameters
    length_threshold: int = 60
    seed_vp_ransac: int = 42
    # Height calculation parameters
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


MIN_FOCAL_LENGTH = 1
MAX_FOCAL_LENGTH = 5000
DEFAULT_FOCAL_LENGTH = 90
DEFAULT_LINE_SCORE_THRESHOLD = 0.1



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
    2. Loads the image and metadata from `height_estimation_parameters.pictures_directory`.
    3. Computes material percentages from the image.
    4. Calls `correction_model.predict` with the raw height and extracted features.
    """
    from imageable.models.height_correction_model import HeightCorrectionModel
    if correction_model is None:
        correction_model = HeightCorrectionModel()
    correction_model.load_model()

    correction_model.pretrained.feature_indices_used_for_clustering = list(range(0,13))

    # 1. Raw height estimate
    raw_height = building_height_from_single_view(height_estimation_parameters)

    # 2–3. Try to load image + metadata and compute material percentages
    street_view_image = None
    material_percentages = None

    pictures_dir = Path(height_estimation_parameters.pictures_directory)

    try:
        image_path = pictures_dir / "image.jpg"
        metadata_path = pictures_dir / "metadata.json"

        street_view_image = np.array(Image.open(image_path))

        with open(metadata_path) as f:
            metadata = json.load(f)

        camera_parameters_dict = metadata["camera_parameters"]
        camera_parameters = CameraParameters(
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
        bmp.camera_parameters = camera_parameters

        material_percentages = get_building_materials_segmentation(bmp)

    except Exception:
        street_view_image = None
        material_percentages = None

    # 4. Corrected height from the model
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



def building_height_from_single_view(
    height_estimation_params: HeightEstimationParameters,
) -> float | None:
    """
    Estimate the height of a building from a single street-level image.

    Parameters
    ----------
    height_estimation_params
        An instance of HeightEstimationParameters containing all necessary parameters.

    Returns
    -------
    height
        Estimated height of the building in meters.
    """
    from imageable.models.line_param_selection_model import LineParameterSelectionModel


    pictures_directory = Path(height_estimation_params.pictures_directory)
    cached_image_path = pictures_directory / "image.jpg"
    cached_metadata_path = pictures_directory / "metadata.json"

    used_cache = False
    if (
        not height_estimation_params.overwrite_images
        and cached_image_path.exists()
        and cached_metadata_path.exists()
    ):
        try:
            image = np.array(Image.open(cached_image_path))
            with cached_metadata_path.open("r") as f:
                metadata_dict = json.load(f)
            cam_dict = metadata_dict.get("camera_parameters", {})
            camera_params = GSVCameraParameters(
                longitude=cam_dict["longitude"],
                latitude=cam_dict["latitude"],
                fov=cam_dict.get("fov", 90),
                heading=cam_dict.get("heading", 0),
                pitch=cam_dict.get("pitch", 0),
                width=cam_dict.get("width", 640),
                height=cam_dict.get("height", 640),
            )
            print("Recycling image and camera parameters from previous run.")
            # Then skip straight to segmentation & height calc using `image` and `camera_params`
            # (everything below in your function stays the same, starting from "segmentation_model = ...")
            used_cache = True

        except Exception:
            # If loading fails, fall back to the normal flow
            pass

    # Image retrieval and camera parameter refinement
    if(not used_cache):
        camera_parameters_refiner = CameraParametersRefiner(height_estimation_params.building_polygon)
        camera_parameters_refiner.MIN_FLOOR_RATIO = height_estimation_params.min_floor_ratio
        camera_parameters_refiner.MIN_SKY_RATIO = height_estimation_params.min_sky_ratio

        camera_params, _success, image = camera_parameters_refiner.adjust_parameters(
            height_estimation_params.gsv_api_key,
            pictures_directory=height_estimation_params.pictures_directory,
            save_reel=height_estimation_params.save_reel,
            overwrite_images=height_estimation_params.overwrite_images,
            confidence_detection=height_estimation_params.confidence_detection,
            max_number_of_images=height_estimation_params.max_number_of_images,
            polygon_buffer_constant=height_estimation_params.polygon_buffer_constant,
        )

    # Building segmentation
    segmentation_model = SegformerSegmentationWrapper(
        model_name=height_estimation_params.segformer_name,
        device=height_estimation_params.device_seg,
        palette_path=height_estimation_params.palette_path,
    )
    segmentation_model.load_model()
    if image is None:
        return None
    results = segmentation_model.predict(image)
    if height_estimation_params.shift_segmentation_labels:
        results = results.squeeze().astype("uint8") + 1
    # Remap the labels
    remapped_seg = segmentation_model._remap_labels(results, height_estimation_params.remapping_dict)
    lcnn_model = LCNNWrapper(
        config_path=height_estimation_params.lcnn_config_path,
        device=height_estimation_params.device_lcnn,
        checkpoint_path=height_estimation_params.lcnn_checkpoint_path,
    )
    # Detect lines in the image
    lcnn_model.load_model()
    lcnn_results = lcnn_model.predict(image)

    # Detect the vanishing points
    vpts_model = VPTSWrapper()

    vpts_dictionary = vpts_model.predict(
        image,
        FOV=camera_params.fov,
        seed=height_estimation_params.seed_vp_ransac,
        length_threshold=height_estimation_params.length_threshold,
    )
    # Predict the best line detection parameter
    line_parameter_selection_model = LineParameterSelectionModel()
    line_parameter_selection_model.load_model()
    
    props = extract_building_properties(
        building_id=height_estimation_params.building_label,
        polygon=height_estimation_params.building_polygon,
        all_buildings=height_estimation_params.all_buildings,
        height_value=None,
        verbose=False,
    )

    props = props.to_dict()
    features_to_use = line_parameter_selection_model.corr_model.FEATURES_USED
    vector = np.array([[props[f] for f in features_to_use]])
    vector = vector.reshape(1,-1)
    try:
        line_score_threshold = line_parameter_selection_model.predict(vector)
        print(f"Predicted line score threshold: {line_score_threshold}")
    except Exception:
        line_score_threshold = DEFAULT_LINE_SCORE_THRESHOLD
    
    config_calculation = {
        "STREET_VIEW": {"HVFoV": str(camera_params.fov), "CameraHeight": "2.5"},
        "SEGMENTATION": {
            "SkyLabel": ",".join(map(str, height_estimation_params.sky_label)),
            "BuildingLabel": ",".join(map(str, height_estimation_params.building_label)),
            "GroundLabel": ",".join(map(str, height_estimation_params.ground_label)),
        },
        "LINE_CLASSIFY": {
            "AngleThres": str(height_estimation_params.line_classification_angle_threshold),
            "LineScore": str(line_score_threshold),
        },
        "LINE_REFINE": {"Edge_Thres": height_estimation_params.edge_threshold},
        "HEIGHT_MEAS": {"MaxDBSANDist": str(height_estimation_params.max_dbscan_distance)},
    }

    height_calculator = HeightCalculator(config_calculation)
    height_estimation_input = HeightEstimationInput(
        image=image,
        segmentation=remapped_seg,
        vanishing_points=vpts_dictionary["vpts_2d"],
        lines=lcnn_results["processed_lines"],
        line_scores=lcnn_results["processed_scores"],
    )

    _height, width = image.shape[:2]
    focal_length = 0.5 * width / np.tan(0.5 * np.radians(camera_params.fov))
    if focal_length < MIN_FOCAL_LENGTH or focal_length > MAX_FOCAL_LENGTH:
        focal_length = DEFAULT_FOCAL_LENGTH

    camera_params_2 = CameraParameters(
        focal_length=focal_length,
        cx=image.shape[1] / 2,
        cy=image.shape[0] / 2,
    )

    results = height_calculator.calculate_heights(
        data=height_estimation_input,
        camera=camera_params_2,
        verbose=height_estimation_params.verbose,
        pitch=camera_params.pitch,
        use_pitch_only=height_estimation_params.use_pitch_only,
        use_detected_vpt_only=height_estimation_params.use_detected_vpt_only,
    )

    #Add correction method.
    if results is None:
        return None
    return mean_no_outliers(collect_heights(results))


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
