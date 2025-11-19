import numpy as np


class SingleViewMetrology:
    """Pure mathematical functions for height calculation"""

    # ruff: noqa PLR0913
    @staticmethod
    def sv_measurement(
        v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, x1: np.ndarray, x2: np.ndarray, zc: float = 2.5
    ) -> float:
        """
        Calculate height using three vanishing points method.

        Parameters
        ----------
        v1
            First vanishing point as a numpy array.
        v2
            Second vanishing point as a numpy array.
        v3
            Third vanishing point as a numpy array.
        x1
            First point on the building edge as a numpy array.
        x2
            Second point on the building edge as a numpy array.
        zc
            Reference height in meters (2.5 m by default).

        Returns
        -------
        zx
            Estimated height in meters.
        """
        vline = np.cross(v1, v2)
        p4 = vline / np.linalg.norm(vline)

        zc = zc * np.linalg.det([v1, v2, v3])
        alpha = -np.linalg.det([v1, v2, p4]) / zc
        p3 = alpha * v3

        zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * np.linalg.norm(np.cross(p3, x2)))
        return abs(zx)

    @staticmethod
    def sv_measurement1(v, vline, x1, x2, zc=2.5):
        p4 = vline / np.linalg.norm(vline)
        alpha = -1 / (np.dot(p4, v) * zc)
        p3 = alpha * v

        zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * np.linalg.norm(np.cross(p3, x2)))
        zx = abs(zx)

        return zx

    @staticmethod
    def calculate_with_cross_ratio(hori_v1, hori_v2, vert_v1, pt_top, pt_bottom, zc=2.5):
        """
        Cross-ratio based height calculation from heightMeasurement.py lines 144-166
        """
        from .processors.line_classifier import LineClassifier  # Import for helper functions

        line_vl = LineClassifier.line_coeff(hori_v1, hori_v2)
        line_building_vert = LineClassifier.line_coeff(pt_top, pt_bottom)
        C = LineClassifier.intersection(line_vl, line_building_vert)

        dist_AC = np.linalg.norm(np.asarray([vert_v1 - C]))
        dist_AB = np.linalg.norm(np.asarray([vert_v1 - pt_top]))
        dist_BD = np.linalg.norm(np.asarray([pt_top - pt_bottom]))
        dist_CD = np.linalg.norm(np.asarray([C - pt_bottom]))

        height = dist_BD * dist_AC / (dist_CD * dist_AB) * zc
        return height

    @staticmethod
    def calculate_with_cross_ratio_vl(hori_vline, vert_v1, pt_top, pt_bottom, zc=2.5):
        """
        Cross-ratio with vanishing line from heightMeasurement.py lines 169-192
        """
        from .processors.line_classifier import LineClassifier

        line_vl = hori_vline
        line_building_vert = LineClassifier.line_coeff(pt_top, pt_bottom)
        C = LineClassifier.intersection(line_vl, line_building_vert)

        dist_AC = np.linalg.norm(np.asarray([vert_v1 - C]))
        dist_AB = np.linalg.norm(np.asarray([vert_v1 - pt_top]))
        dist_BD = np.linalg.norm(np.asarray([pt_top - pt_bottom]))
        dist_CD = np.linalg.norm(np.asarray([C - pt_bottom]))

        height = dist_BD * dist_AC / (dist_CD * dist_AB) * zc
        return height
