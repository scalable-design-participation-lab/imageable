"""
Image-based feature extraction.

This module provides functions and classes to extract color, shape,
and façade features from building images.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ImageCalculator:
    """
    Calculator for extracting features from building images.

    This class stores the masked building image to avoid redundant calculations
    and provides methods for extracting various visual features.
    """

    def __init__(self, img: np.ndarray, building_mask: np.ndarray | None = None):
        """
        Initialize the calculator with an image and optional building mask.

        Args:
            img: RGB image as numpy array (H, W, 3)
            building_mask: Binary mask of building pixels (H, W). If None,
                          assumes entire image is the building.
        """
        self.original_img = img

        if building_mask is None:
            self.building_mask = np.ones((img.shape[0], img.shape[1]), dtype=bool)
        else:
            self.building_mask = building_mask.astype(bool)

        # Apply mask to image
        self.masked_img = img.copy()
        for channel in range(3):
            self.masked_img[:, :, channel] = img[:, :, channel] * self.building_mask

        # Convert to HSV for brightness and vividness calculations
        self.hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        self.masked_hsv = self.hsv_img.copy()
        for channel in range(3):
            self.masked_hsv[:, :, channel] = self.hsv_img[:, :, channel] * self.building_mask

    # ========== Color Features ==========

    def average_red_channel_value(self) -> float:
        """Calculate average red channel value in building mask."""
        masked_values = self.original_img[:, :, 0][self.building_mask]
        return float(np.mean(masked_values)) if len(masked_values) > 0 else 0.0

    def average_green_channel_value(self) -> float:
        """Calculate average green channel value in building mask."""
        masked_values = self.original_img[:, :, 1][self.building_mask]
        return float(np.mean(masked_values)) if len(masked_values) > 0 else 0.0

    def average_blue_channel_value(self) -> float:
        """Calculate average blue channel value in building mask."""
        masked_values = self.original_img[:, :, 2][self.building_mask]
        return float(np.mean(masked_values)) if len(masked_values) > 0 else 0.0

    def average_brightness(self) -> float:
        """
        Calculate average brightness as percentage.

        Brightness is extracted from the V channel in HSV color space.

        Returns
        -------
            Brightness as percentage (0-100)
        """
        masked_values = self.hsv_img[:, :, 2][self.building_mask]
        if len(masked_values) == 0:
            return 0.0
        avg = np.mean(masked_values)
        per = (avg / 255) * 100
        return float(per)

    def average_vividness(self) -> float:
        """
        Calculate average vividness (saturation) as percentage.

        Vividness is extracted from the S channel in HSV color space.

        Returns
        -------
            Vividness as percentage (0-100)
        """
        masked_values = self.hsv_img[:, :, 1][self.building_mask]
        if len(masked_values) == 0:
            return 0.0
        avg = np.mean(masked_values)
        per = (avg / 255) * 100
        return float(per)

    # ========== Shape and Geometry Features ==========

    def mask_area(self) -> int:
        """Number of pixels occupied by building."""
        return int(np.sum(self.building_mask))

    def mask_length(self) -> float:
        """
        Length of the polygon enclosing the segmented building (in pixels).

        Uses contour detection to find the building boundary.
        """
        # Find contours in the mask
        mask_uint8 = self.building_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return 0.0

        # Get the largest contour (main building boundary)
        largest_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest_contour, closed=True)

        return float(perimeter)

    def mask_complexity(self) -> float:
        """
        Complexity ratio: mask_length / mask_area.

        Higher values indicate more complex/irregular shapes.
        """
        area = self.mask_area()
        if area == 0:
            return 0.0
        length = self.mask_length()
        return float(length / area)

    def number_of_edges(self) -> int:
        """
        Number of edges in the polygon that encloses the building mask.

        Uses polygon approximation to reduce contour to significant edges.
        """
        mask_uint8 = self.building_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return 0

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        return len(approx)

    def number_of_vertices(self) -> int:
        """
        Number of vertices in the polygon that encloses the building mask.

        This is equivalent to number_of_edges for closed polygons.
        """
        return self.number_of_edges()

    # ========== Façade Features ==========

    def _compute_center_of_mass(self, mask: np.ndarray) -> tuple[float, float]:
        """
        Compute center of mass for a binary mask.

        Args:
            mask: Binary mask array

        Returns
        -------
            (x, y) coordinates of center of mass
        """
        if np.sum(mask) == 0:
            return (0.0, 0.0)

        y_coords, x_coords = np.where(mask)
        x_center = np.mean(x_coords)
        y_center = np.mean(y_coords)

        return (float(x_center), float(y_center))

    def average_window_x(self, window_mask: np.ndarray) -> float:
        """
        Calculate average horizontal location of windows.

        Args:
            window_mask: Binary mask of window pixels

        Returns
        -------
            Average x coordinate
        """
        x, _ = self._compute_center_of_mass(window_mask)
        return x

    def average_window_y(self, window_mask: np.ndarray) -> float:
        """
        Calculate average vertical location of windows.

        Args:
            window_mask: Binary mask of window pixels

        Returns
        -------
            Average y coordinate
        """
        _, y = self._compute_center_of_mass(window_mask)
        return y

    def average_door_x(self, door_mask: np.ndarray) -> float:
        """Calculate average horizontal location of doors."""
        x, _ = self._compute_center_of_mass(door_mask)
        return x

    def average_door_y(self, door_mask: np.ndarray) -> float:
        """Calculate average vertical location of doors."""
        _, y = self._compute_center_of_mass(door_mask)
        return y

    def _optimal_clusters(self, mask: np.ndarray, max_k: int = 10) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            mask: Binary mask of features (windows or doors)
            max_k: Maximum number of clusters to try

        Returns
        -------
            Optimal number of clusters
        """
        # Get coordinates of mask pixels
        y_coords, x_coords = np.where(mask)

        if len(x_coords) < 2:
            return 0 if len(x_coords) == 0 else 1

        # Create feature matrix
        coords = np.column_stack([x_coords, y_coords])

        # Try different k values
        best_k = 1
        best_score = -1

        for k in range(2, min(max_k + 1, len(coords))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(coords)

            # Calculate silhouette score
            score = silhouette_score(coords, labels)

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def number_of_windows(self, window_mask: np.ndarray, max_k: int = 20) -> int:
        """
        Count number of distinct windows using clustering.

        Args:
            window_mask: Binary mask of window pixels
            max_k: Maximum number of windows to search for

        Returns
        -------
            Optimal number of window clusters
        """
        return self._optimal_clusters(window_mask, max_k)

    def number_of_doors(self, door_mask: np.ndarray, max_k: int = 10) -> int:
        """
        Count number of distinct doors using clustering.

        Args:
            door_mask: Binary mask of door pixels
            max_k: Maximum number of doors to search for

        Returns
        -------
            Optimal number of door clusters
        """
        return self._optimal_clusters(door_mask, max_k)

    # ========== Convenience Method ==========

    def extract_all_features(
        self, window_mask: np.ndarray | None = None, door_mask: np.ndarray | None = None
    ) -> dict[str, float]:
        """
        Extract all available features.

        Args:
            window_mask: Optional binary mask of windows for façade features
            door_mask: Optional binary mask of doors for façade features

        Returns
        -------
            Dictionary of feature names and values
        """
        features = {
            # Color features
            "average_red_channel_value": self.average_red_channel_value(),
            "average_green_channel_value": self.average_green_channel_value(),
            "average_blue_channel_value": self.average_blue_channel_value(),
            "average_brightness": self.average_brightness(),
            "average_vividness": self.average_vividness(),
            # Shape features
            "mask_area": self.mask_area(),
            "mask_length": self.mask_length(),
            "mask_complexity": self.mask_complexity(),
            "number_of_edges": self.number_of_edges(),
            "number_of_vertices": self.number_of_vertices(),
        }

        # Add façade features if masks provided
        if window_mask is not None:
            features["average_window_x"] = self.average_window_x(window_mask)
            features["average_window_y"] = self.average_window_y(window_mask)
            features["number_of_windows"] = self.number_of_windows(window_mask)

        if door_mask is not None:
            features["average_door_x"] = self.average_door_x(door_mask)
            features["average_door_y"] = self.average_door_y(door_mask)
            features["number_of_doors"] = self.number_of_doors(door_mask)

        return features
