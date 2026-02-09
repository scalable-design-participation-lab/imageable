from imageable._models.base import BaseModelWrapper
from imageable._correction_ensembles.cluster_weighted_ensemble import (
    ClusterWeightedEnsembleWrapper,
    ClusterWeightedEnsembleSpatialWrapper,
)
from huggingface_hub import hf_hub_download, try_to_load_from_cache
import joblib
from imageable._extraction.extract import extract_building_properties
from imageable._features.height.building_height import HeightEstimationParameters
from shapely.geometry import Polygon
import numpy as np
from typing import ClassVar


class HeightCorrectionModel(BaseModelWrapper):

    # Features used for clustering / correction (in this order)
    FEATURES_USED = [
        "unprojected_area",
        "projected_area",
        "longitude_difference",
        "latitude_difference",
        "n_vertices",
        "shape_length",
        "complexity",
        'inverse_average_segment_length',
        'vertices_per_area',
        'average_complexity_per_segment',
        "isoperimetric_quotient",
        "mean_distance_to_neighbors",
        "nearest_neighbor_distance",
        "building_height"
    ]

    FEATURES_MAPPING_DICT: ClassVar[dict[str, int]] = {
        "unprojected_area": 0,
        "projected_area": 1,
        "longitude_difference": 2,
        "latitude_difference": 3,
        "n_vertices": 4,
        "shape_length": 5,
        "complexity": 6,
        "inverse_average_segment_length": 7,
        "vertices_per_area": 8,
        "average_complexity_per_segment": 9,
        "isoperimetric_quotient": 10,
        "mean_distance_to_neighbors": 12,
        "nearest_neighbor_distance": 14,
        "building_height": 21,
    }

    MODEL_REPO = "walup/cluster_height_correction_model"
    MODEL_FILE_NAME = "cluster_ensemble_model.pkl"
    SCALER_FILE_NAME = "cluster_ensemble_scaler.pkl"

    def __init__(
        self,
        pretrained: ClusterWeightedEnsembleWrapper | ClusterWeightedEnsembleSpatialWrapper = None,
    ) -> None:
        self.pretrained = pretrained
        self.scaler = None

    def _download_model(self) -> None:
        # --- 1) Ensemble model ---
        cached = try_to_load_from_cache(
            repo_id=self.MODEL_REPO,
            filename=self.MODEL_FILE_NAME,
        )

        if cached is None:
            model_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.MODEL_FILE_NAME,
            )
        else:
            model_path = cached

        if model_path is None:
            raise ValueError("Could not download model from huggingface")

        self.pretrained = joblib.load(model_path)

        cached_scaler = try_to_load_from_cache(
            repo_id=self.MODEL_REPO,
            filename=self.SCALER_FILE_NAME,
        )

        if cached_scaler is None:
            scaler_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.SCALER_FILE_NAME,
            )
        else:
            scaler_path = cached_scaler

        if scaler_path is None:
            raise ValueError("Could not download scaler from huggingface")

        self.scaler = joblib.load(scaler_path)

    def load_model(self) -> None:
        if self.pretrained is None or not self.pretrained.is_loaded():
            self._download_model()

    def is_loaded(self) -> bool:
        return (self.pretrained is not None) and self.pretrained.is_loaded()

    def preprocess(self, image):
        return super().preprocess(image)

    def postprocess(self, outputs):
        return super().postprocess(outputs)

    def predict(
        self,
        raw_height: float,
        estimation_params: HeightEstimationParameters,
        building_id: int,
        all_buildings: list[Polygon],
        crs: str = "EPSG:4326",
        street_view_image=None,
        material_percentages: dict | None = None,
        verbose: bool = False,
    ) -> float:
        # For extracting we need (1) the building id, (2) the polygon,
        # (3) other buildings in the area, (4) the building image (optional),
        # (5) material percentages (optional).

        if not self.is_loaded():
            raise ValueError("Model not loaded. Please load the model before using this method.")

        if self.scaler is None:
            raise RuntimeError("Scaler not loaded. Make sure the scaler file is available and loaded.")

        # Get the footprint for this building
        footprint = estimation_params.building_polygon

        properties = extract_building_properties(
            building_id=building_id,
            polygon=footprint,
            all_buildings=all_buildings,
            crs=crs,
            street_view_image=street_view_image,
            height_value=raw_height,
            material_percentages=material_percentages,
            verbose=verbose,
        )

        properties_dictionary = properties.to_dict()

        # Build raw feature vector in the same order as training
        x_raw = np.array([[properties_dictionary[f] for f in self.FEATURES_USED]])
        #print(x_raw)

        # Scale using the SAME scaler used when you created X_scaled in training
        x_scaled = self.scaler.transform(x_raw)
        #print(x_scaled)

        # Predict the corrected height (ensemble was trained on scaled features)
        corrected_height = self.pretrained.predict(x_scaled)[0]
        #print(self.pretrained.cluster_centers_)
        return corrected_height

    


