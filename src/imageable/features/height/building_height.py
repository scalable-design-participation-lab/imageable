
from shapely import Polygon
from imageable.images.camera.camera_adjustment import CameraParametersRefiner
from typing import Tuple, List,Dict
from imageable.models.huggingface.segformer_segmentation import SegformerSegmentationWrapper
from PIL import Image
import numpy as np
from imageable.models.lcnn.lcnn_wrapper import LCNNWrapper
from imageable.models.vpts.vpts_wrapper import VPTSWrapper
from imageable.models.height_estimation.height_calculator import HeightCalculator, CameraParameters, HeightEstimationInput
from pathlib import Path

class HeightEstimationParameters:
    
    
    def __init__(self,
        gsv_api_key:str, 
        building_polygon:Polygon,
        pictures_directory:Path = str(Path(__file__).resolve().parents[3] / "notebooks" / "pictures"),
        #Parameter refinement parameters
        save_reel:bool = False,
        overwrite_images:bool = True,
        confidence_detection:float = 0.1,
        max_number_of_images:int = 5,
        polygon_buffer_constant = 1.5e5,
        min_floor_ratio:float = 0.00001,
        min_sky_ratio:float = 0.1,
        #Building Segmentation parameters
        segformer_name:str = "nvidia/segformer-b5-finetuned-ade-640-640",
        palette_path:str = str((Path(__file__).resolve().parents[2] / "assets" / "ade20k_palette.json")),
        device_seg:str = "cpu",
        remapping_dict:Dict[int, int] = {
            2: 1,
            26: 1,
            3: 2,
            12: 11,
            7: 11,
            4: 11,
            10:11,
            14:11
        },
        #Line segmentation parameters
        lcnn_config_path:str = str(Path(__file__).resolve().parents[2] / "assets" / "wireframe.yaml"),
        device_lcnn:str = "cpu",
        lcnn_checkpoint_path:str = "",
        #Vanishing point parameters
        length_threshold:int = 60,
        seed_vp_ransac:int = 42,
        #Height calculation parameters
        sky_label:List[int] = [0,2], 
        building_label:List[int] = [1],
        ground_label:List[int]  = [6,11],
        line_classification_angle_threshold:float = 10.0,
        line_score_threshold:float = 0.94,
        edge_thres: str = "5,5",
        max_dbscan_distance:float = 60.0,
        verbose:bool = False,
        use_pitch_only:bool = False,
        use_detected_vpt_only:bool = False,
        shift_segmentation_labels:bool = True
        
):
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
        self.gsv_api_key = gsv_api_key
        self.building_polygon = building_polygon
        self.pictures_directory = pictures_directory
        #Parameter refinement parameters
        self.save_reel = save_reel
        self.overwrite_images = overwrite_images
        self.confidence_detection = confidence_detection
        self.max_number_of_images = max_number_of_images
        self.polygon_buffer_constant = polygon_buffer_constant
        self.min_floor_ratio = min_floor_ratio
        self.min_sky_ratio = min_sky_ratio
        #Building Segmentation parameters
        self.segformer_name = segformer_name
        self.palette_path = palette_path
        self.device_seg = device_seg
        self.remapping_dict = remapping_dict
        #Line segmentation parameters
        self.lcnn_config_path = lcnn_config_path
        self.device_lcnn = device_lcnn
        self.lcnn_checkpoint_path = lcnn_checkpoint_path
        #Vanishing point parameters
        self.length_threshold = length_threshold
        self.seed_vp_ransac = seed_vp_ransac
        #Height calculation parameters
        self.sky_label = sky_label
        self.building_label = building_label
        self.ground_label = ground_label
        self.line_classification_angle_threshold = line_classification_angle_threshold
        self.line_score_threshold = line_score_threshold
        self.edge_threshold = edge_thres
        self.max_dbscan_distance = max_dbscan_distance
        self.verbose = verbose
        self.use_pitch_only = use_pitch_only
        self.use_detected_vpt_only = use_detected_vpt_only
        self.shift_segmentation_labels = shift_segmentation_labels
    
    
    

def building_height_from_single_view(
    height_estimation_params: HeightEstimationParameters,
) -> float:
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
    #Image retrieval and camera parameter refinement
    camera_parameters_refiner = CameraParametersRefiner(height_estimation_params.building_polygon)
    camera_parameters_refiner.MIN_FLOOR_RATIO = height_estimation_params.min_floor_ratio
    camera_parameters_refiner.MIN_SKY_RATIO = height_estimation_params.min_sky_ratio
    
    camera_params, success, image = camera_parameters_refiner.adjust_parameters(
    height_estimation_params.gsv_api_key,
    pictures_directory = height_estimation_params.pictures_directory,
    save_reel = height_estimation_params.save_reel,
    overwrite_images = height_estimation_params.overwrite_images,
    confidence_detection = height_estimation_params.confidence_detection,
    max_number_of_images = height_estimation_params.max_number_of_images,
    polygon_buffer_constant = height_estimation_params.polygon_buffer_constant
    )
    
    #Building segmentation
    segmentation_model = SegformerSegmentationWrapper(
    model_name=height_estimation_params.segformer_name,
    device=height_estimation_params.device_seg,
    palette_path=height_estimation_params.palette_path
    )
    segmentation_model.load_model()
    results = segmentation_model.predict(image)
    if(height_estimation_params.shift_segmentation_labels):
        results= results.squeeze().astype('uint8') + 1
    #Remap the labels 
    remapped_seg = segmentation_model._remap_labels(results, height_estimation_params.remapping_dict)
    lcnn_model = LCNNWrapper(
        config_path = height_estimation_params.lcnn_config_path,
        device = height_estimation_params.device_lcnn,
        checkpoint_path = height_estimation_params.lcnn_checkpoint_path
    )
    #Detect lines in the image
    lcnn_model.load_model()
    lcnn_results = lcnn_model.predict(image)
    
    #Detect the vanishing points
    vpts_model = VPTSWrapper()

    vpts_dictionary = vpts_model.predict(
        image, 
        FOV = camera_params.fov,
        seed = height_estimation_params.seed_vp_ransac,
        length_threshold = height_estimation_params.length_threshold)
    config_calculation = {"STREET_VIEW": {
            "HVFoV": str(camera_params.fov),
            "CameraHeight": "2.5"
        },
        "SEGMENTATION": {
            "SkyLabel": ",".join(map(str, height_estimation_params.sky_label)),
            "BuildingLabel": ",".join(map(str, height_estimation_params.building_label)),
            "GroundLabel": ",".join(map(str, height_estimation_params.ground_label))
        },
        "LINE_CLASSIFY": {
            "AngleThres": str(height_estimation_params.line_classification_angle_threshold),
            "LineScore": str(height_estimation_params.line_score_threshold)
        },
        "LINE_REFINE": {
            "Edge_Thres": height_estimation_params.edge_threshold
        },
        "HEIGHT_MEAS": {
            "MaxDBSANDist": str(height_estimation_params.max_dbscan_distance)
        }}

    height_calculator = HeightCalculator(config_calculation)
    height_estimation_input = HeightEstimationInput(
    image = image,
    segmentation = remapped_seg,
    vanishing_points= vpts_dictionary["vpts_2d"],
    lines = lcnn_results["processed_lines"],
    line_scores = lcnn_results["processed_scores"],
    )
    
    height, width = image.shape[:2]
    focal_length = 0.5 * width / np.tan(0.5 * np.radians(camera_params.fov))
    if(focal_length < 1 or focal_length > 5000):
        focal_length = 90
    
    camera_params_2 = CameraParameters(
        focal_length = focal_length,
        cx= image.shape[1] / 2,
        cy= image.shape[0] / 2,
    )
    
    results = height_calculator.calculate_heights(
        data = height_estimation_input,
        camera = camera_params_2,
        verbose=height_estimation_params.verbose,
        pitch=camera_params.pitch,
        use_pitch_only=height_estimation_params.use_pitch_only,
        use_detected_vpt_only=height_estimation_params.use_detected_vpt_only
    )
    return mean_no_outliers(collect_heights(results))

def collect_heights(results: Dict) -> List[float]:
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
    all_heights = [
    line[0]                       # the height value
    for building in results["heights"]
    for line in building["lines"] # each line in that building
    ]
    return all_heights

def mean_no_outliers(values: List[float]) -> float:
    values = np.array(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = values[(values >= lower) & (values <= upper)]
    return np.mean(filtered)

    
    


    
    
        
    
    
    
    

    
    
    
