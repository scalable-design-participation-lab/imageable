"""
Comprehensive building feature extraction class.

This module integrates all building features: footprint, height, materials, and image features.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class BuildingProperties:
    """
    Comprehensive container for all building features.

    This class encapsulates features from multiple sources:
    - Footprint features (geometric, engineered, contextual)
    - Height estimation
    - Material percentages from segmentation
    - Image-based features (color, shape, façade)
    """

    # ========== Footprint: Geometrical Features ==========
    unprojected_area: float = 0.0
    projected_area: float = 0.0
    longitude_difference: float = 0.0
    latitude_difference: float = 0.0
    n_vertices: int = 0
    shape_length: float = 0.0

    # ========== Footprint: Engineered Features ==========
    complexity: float = 0.0
    inverse_average_segment_length: float = 0.0
    vertices_per_area: float = 0.0
    average_complexity_per_segment: float = 0.0
    isoperimetric_quotient: float = 0.0

    # ========== Footprint: Contextual Features ==========
    neighbor_count: int = 0
    mean_distance_to_neighbors: float = 0.0
    expected_nearest_neighbor_distance: float = 0.0
    nearest_neighbor_distance: float = 0.0
    n_size_mean: float = 0.0
    n_size_std: float = 0.0
    n_size_min: float = 0.0
    n_size_max: float = 0.0
    n_size_cv: float = 0.0
    nni: float = 0.0  # Nearest neighbor index

    # ========== Height Estimation ==========
    building_height: float = -1.0  # -1 indicates height could not be calculated

    # ========== Material Percentages ==========
    material_percentages: dict[str, float] = field(default_factory=dict)

    # ========== Image: Color Features ==========
    average_red_channel_value: float = 0.0
    average_green_channel_value: float = 0.0
    average_blue_channel_value: float = 0.0
    average_brightness: float = 0.0
    average_vividness: float = 0.0

    # ========== Image: Shape Features ==========
    mask_area: int = 0
    mask_length: float = 0.0
    mask_complexity: float = 0.0
    number_of_edges: int = 0
    number_of_vertices: int = 0

    # ========== Image: Façade Features ==========
    average_window_x: float = 0.0
    average_window_y: float = 0.0
    average_door_x: float = 0.0
    average_door_y: float = 0.0
    number_of_windows: int = 0
    number_of_doors: int = 0

    # ========== Metadata ==========
    building_id: str | None = None

    def update_footprint_features(self, footprint_dict: dict[str, Any]) -> None:
        """
        Update footprint features from a dictionary.

        Args:
            footprint_dict: Dictionary containing footprint features
        """
        # Geometrical features
        if "unprojected_area" in footprint_dict:
            self.unprojected_area = float(footprint_dict["unprojected_area"])
        if "projected_area" in footprint_dict:
            self.projected_area = float(footprint_dict["projected_area"])
        if "longitude_difference" in footprint_dict:
            self.longitude_difference = float(footprint_dict["longitude_difference"])
        if "latitude_difference" in footprint_dict:
            self.latitude_difference = float(footprint_dict["latitude_difference"])
        if "n_vertices" in footprint_dict:
            self.n_vertices = int(footprint_dict["n_vertices"])
        if "shape_length" in footprint_dict:
            self.shape_length = float(footprint_dict["shape_length"])

        # Engineered features
        if "complexity" in footprint_dict:
            self.complexity = float(footprint_dict["complexity"])
        if "inverse_average_segment_length" in footprint_dict:
            self.inverse_average_segment_length = float(footprint_dict["inverse_average_segment_length"])
        if "vertices_per_area" in footprint_dict:
            self.vertices_per_area = float(footprint_dict["vertices_per_area"])
        if "average_complexity_per_segment" in footprint_dict:
            self.average_complexity_per_segment = float(footprint_dict["average_complexity_per_segment"])
        if "isoperimetric_quotient" in footprint_dict:
            self.isoperimetric_quotient = float(footprint_dict["isoperimetric_quotient"])

        # Contextual features
        if "neighbor_count" in footprint_dict:
            self.neighbor_count = int(footprint_dict["neighbor_count"])
        if "mean_distance_to_neighbors" in footprint_dict:
            self.mean_distance_to_neighbors = float(footprint_dict["mean_distance_to_neighbors"])
        if "expected_nearest_neighbor_distance" in footprint_dict:
            self.expected_nearest_neighbor_distance = float(footprint_dict["expected_nearest_neighbor_distance"])
        if "nearest_neighbor_distance" in footprint_dict:
            self.nearest_neighbor_distance = float(footprint_dict["nearest_neighbor_distance"])
        if "n_size_mean" in footprint_dict:
            self.n_size_mean = float(footprint_dict["n_size_mean"])
        if "n_size_std" in footprint_dict:
            self.n_size_std = float(footprint_dict["n_size_std"])
        if "n_size_min" in footprint_dict:
            self.n_size_min = float(footprint_dict["n_size_min"])
        if "n_size_max" in footprint_dict:
            self.n_size_max = float(footprint_dict["n_size_max"])
        if "n_size_cv" in footprint_dict:
            self.n_size_cv = float(footprint_dict["n_size_cv"])
        if "nni" in footprint_dict:
            self.nni = float(footprint_dict["nni"])

    def update_height(self, height: float | None) -> None:
        """
        Update building height estimation.

        Args:
            height: Estimated building height, or None if calculation failed
        """
        self.building_height = float(height) if height is not None else -1.0

    def update_material_percentages(self, material_dict: dict[str, float]) -> None:
        """
        Update material percentages from segmentation.

        Args:
            material_dict: Dictionary mapping material names to percentages
        """
        self.material_percentages = material_dict.copy()

    def update_image_features(self, image_features_dict: dict[str, Any]) -> None:
        """
        Update image-based features from a dictionary.

        Args:
            image_features_dict: Dictionary containing image features
        """
        # Color features
        if "average_red_channel_value" in image_features_dict:
            self.average_red_channel_value = float(image_features_dict["average_red_channel_value"])
        if "average_green_channel_value" in image_features_dict:
            self.average_green_channel_value = float(image_features_dict["average_green_channel_value"])
        if "average_blue_channel_value" in image_features_dict:
            self.average_blue_channel_value = float(image_features_dict["average_blue_channel_value"])
        if "average_brightness" in image_features_dict:
            self.average_brightness = float(image_features_dict["average_brightness"])
        if "average_vividness" in image_features_dict:
            self.average_vividness = float(image_features_dict["average_vividness"])

        # Shape features
        if "mask_area" in image_features_dict:
            self.mask_area = int(image_features_dict["mask_area"])
        if "mask_length" in image_features_dict:
            self.mask_length = float(image_features_dict["mask_length"])
        if "mask_complexity" in image_features_dict:
            self.mask_complexity = float(image_features_dict["mask_complexity"])
        if "number_of_edges" in image_features_dict:
            self.number_of_edges = int(image_features_dict["number_of_edges"])
        if "number_of_vertices" in image_features_dict:
            self.number_of_vertices = int(image_features_dict["number_of_vertices"])

        # Façade features
        if "average_window_x" in image_features_dict:
            self.average_window_x = float(image_features_dict["average_window_x"])
        if "average_window_y" in image_features_dict:
            self.average_window_y = float(image_features_dict["average_window_y"])
        if "average_door_x" in image_features_dict:
            self.average_door_x = float(image_features_dict["average_door_x"])
        if "average_door_y" in image_features_dict:
            self.average_door_y = float(image_features_dict["average_door_y"])
        if "number_of_windows" in image_features_dict:
            self.number_of_windows = int(image_features_dict["number_of_windows"])
        if "number_of_doors" in image_features_dict:
            self.number_of_doors = int(image_features_dict["number_of_doors"])

    def to_dict(self) -> dict[str, Any]:
        """
        Convert features to a dictionary.

        Returns
        -------
            Dictionary of all features
        """
        return asdict(self)

    def to_json(self, filepath: str | None = None) -> str:
        """
        Convert features to JSON.

        Args:
            filepath: Optional path to save JSON file

        Returns
        -------
            JSON string
        """
        json_str = json.dumps(self.to_dict(), indent=2)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_dict(cls, feature_dict: dict[str, Any]) -> "BuildingProperties":
        """
        Create BuildingProperties from a dictionary.

        Args:
            feature_dict: Dictionary containing features

        Returns
        -------
            BuildingProperties instance
        """
        return cls(**feature_dict)

    @classmethod
    def from_json(cls, json_str_or_path: str) -> "BuildingProperties":
        """
        Create BuildingProperties from JSON string or file.

        Args:
            json_str_or_path: JSON string or path to JSON file

        Returns
        -------
            BuildingProperties instance
        """
        # Try to read as file first
        try:
            with open(json_str_or_path) as f:
                feature_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Treat as JSON string
            feature_dict = json.loads(json_str_or_path)

        return cls.from_dict(feature_dict)

    def get_feature_vector(self, exclude_materials: bool = False) -> np.ndarray:
        """
        Get features as a numeric vector for ML models.

        Args:
            exclude_materials: If True, exclude material percentages

        Returns
        -------
            NumPy array of feature values
        """
        features = []

        # Footprint geometrical
        features.extend(
            [
                self.unprojected_area,
                self.projected_area,
                self.longitude_difference,
                self.latitude_difference,
                self.n_vertices,
                self.shape_length,
            ]
        )

        # Footprint engineered
        features.extend(
            [
                self.complexity,
                self.inverse_average_segment_length,
                self.vertices_per_area,
                self.average_complexity_per_segment,
                self.isoperimetric_quotient,
            ]
        )

        # Footprint contextual
        features.extend(
            [
                self.neighbor_count,
                self.mean_distance_to_neighbors,
                self.expected_nearest_neighbor_distance,
                self.nearest_neighbor_distance,
                self.n_size_mean,
                self.n_size_std,
                self.n_size_min,
                self.n_size_max,
                self.n_size_cv,
                self.nni,
            ]
        )

        # Height
        features.append(self.building_height)

        # Materials
        if not exclude_materials:
            material_values = sorted(self.material_percentages.items())
            features.extend([val for _, val in material_values])

        # Image color
        features.extend(
            [
                self.average_red_channel_value,
                self.average_green_channel_value,
                self.average_blue_channel_value,
                self.average_brightness,
                self.average_vividness,
            ]
        )

        # Image shape
        features.extend(
            [
                self.mask_area,
                self.mask_length,
                self.mask_complexity,
                self.number_of_edges,
                self.number_of_vertices,
            ]
        )

        # Image façade
        features.extend(
            [
                self.average_window_x,
                self.average_window_y,
                self.average_door_x,
                self.average_door_y,
                self.number_of_windows,
                self.number_of_doors,
            ]
        )

        return np.array(features)

    def get_feature_names(self, exclude_materials: bool = False) -> list[str]:
        """
        Get ordered list of feature names matching get_feature_vector().

        Args:
            exclude_materials: If True, exclude material percentage names

        Returns
        -------
            List of feature names
        """
        names = [
            # Footprint geometrical
            "unprojected_area",
            "projected_area",
            "longitude_difference",
            "latitude_difference",
            "n_vertices",
            "shape_length",
            # Footprint engineered
            "complexity",
            "inverse_average_segment_length",
            "vertices_per_area",
            "average_complexity_per_segment",
            "isoperimetric_quotient",
            # Footprint contextual
            "neighbor_count",
            "mean_distance_to_neighbors",
            "expected_nearest_neighbor_distance",
            "nearest_neighbor_distance",
            "n_size_mean",
            "n_size_std",
            "n_size_min",
            "n_size_max",
            "n_size_cv",
            "nni",
            # Height
            "building_height",
        ]

        # Materials
        if not exclude_materials:
            material_names = sorted(self.material_percentages.keys())
            names.extend([f"material_{name}" for name in material_names])

        # Image features
        names.extend(
            [
                "average_red_channel_value",
                "average_green_channel_value",
                "average_blue_channel_value",
                "average_brightness",
                "average_vividness",
                "mask_area",
                "mask_length",
                "mask_complexity",
                "number_of_edges",
                "number_of_vertices",
                "average_window_x",
                "average_window_y",
                "average_door_x",
                "average_door_y",
                "number_of_windows",
                "number_of_doors",
            ]
        )

        return names

    def __repr__(self) -> str:
        """String representation showing key features."""
        return (
            f"BuildingFeatures(id={self.building_id}, "
            f"area={self.projected_area:.2f}, "
            f"height={self.building_height:.2f}, "
            f"n_features={len(self.get_feature_vector())})"
        )
