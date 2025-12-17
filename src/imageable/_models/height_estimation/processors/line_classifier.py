import copy

import numpy as np
from sklearn.cluster import DBSCAN


class LineClassifier:
    """Line classification based on vanishing points and building area."""

    def classify_with_vpts(self, n1: np.ndarray, n2: np.ndarray, vpt: np.ndarray, config: dict) -> bool:
        """From lineClassification.py lines 15-51."""
        flag = False
        t_angle = float(config["LINE_CLASSIFY"]["AngleThres"])

        p1 = np.array([n1[1], n1[0]])
        p2 = np.array([n2[1], n2[0]])

        mpt = np.array([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0])
        d1 = p2 - p1
        d2 = vpt - mpt

        norm_d1 = np.linalg.norm(d1)
        norm_d2 = np.linalg.norm(d2)

        if norm_d1 == 0 or norm_d2 == 0:
            return False  # degenerate, can't classify

        cos_theta = np.dot(d1, d2) / (norm_d1 * norm_d2)
        # numerical safety: clip to [-1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle = np.rad2deg(np.arccos(cos_theta))

        if angle < t_angle or 180 - angle < t_angle:
            flag = True

        return flag

    def check_if_line_lies_in_building_area(
        self, seg_img: np.ndarray, a: np.ndarray, b: np.ndarray, config: dict
    ) -> bool:
        """From lineClassification.py lines 54-104."""
        middle = (a + b) / 2.0
        if np.allclose(a, b):
            return False
        norm_direction = (a - b) / np.linalg.norm(a - b)
        ppd_dir = np.asarray([norm_direction[1], -norm_direction[0]])

        building_label = int(config["SEGMENTATION"]["BuildingLabel"])
        # Fix np.cast syntax:
        # np.array(config["SEGMENTATION"]["GroundLabel"].split(","), dtype=int)  # noqa: ERA001

        ratio = 10
        ppd_dir = ratio * ppd_dir
        point_check_list = copy.deepcopy(a)
        point_check_list = np.vstack([point_check_list, a - ppd_dir])
        point_check_list = np.vstack([point_check_list, a + ppd_dir])
        point_check_list = np.vstack([point_check_list, b])
        point_check_list = np.vstack([point_check_list, b - ppd_dir])
        point_check_list = np.vstack([point_check_list, b + ppd_dir])
        point_check_list = np.vstack([point_check_list, middle])
        point_check_list = np.vstack([point_check_list, middle - ppd_dir])
        point_check_list = np.vstack([point_check_list, middle + ppd_dir])
        point_check_list = [v for v in point_check_list if not np.isnan(v).any()]

        total_num = 0
        local_num = 0
        rows, cols = seg_img.shape
        flag = True

        for pcl in point_check_list:
            total_num = total_num + 1
            y_int = int(pcl[0] + 0.5)
            x_int = int(pcl[1] + 0.5)

            if x_int < 0 or x_int > cols - 1 or y_int < 0 or y_int > rows - 1:
                local_num = local_num + 1
                continue
            if seg_img[y_int, x_int] == building_label:
                local_num = local_num + 1
            if np.remainder(total_num, 3) == 0 and local_num == 0:
                flag = False
                break
            if np.remainder(total_num, 3) == 0:
                local_num = 0
        return flag

    def cluster_lines_with_centers(self, ht_set: list, config: dict, using_height: bool = False) -> list | None:
        """From lineClassification.py lines 436-478."""
        feature_points = []
        if using_height:
            for ht, a, b, *_ in ht_set:
                feature_points.append([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, ht])
        else:
            for _, a, b, *_ in ht_set:
                feature_points.append([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
        feature_points = np.asarray(feature_points)

        max_dbscan_dist = float(config["HEIGHT_MEAS"]["MaxDBSANDist"])

        try:
            clustering = DBSCAN(eps=max_dbscan_dist, min_samples=1).fit(feature_points)
        except (ValueError, KeyError) as e:
            print(f"Error in clustering {e}")
            return None

        clustered_lines = []
        max_val = np.max(clustering.labels_) + 1
        for label in range(max_val):
            new_list = []
            new_ht_list = []
            for i in range(len(clustering.labels_)):
                if clustering.labels_[i] == label:
                    new_list.append(ht_set[i])
                    new_ht_list.append(ht_set[i][0])
            medi_val = np.median(np.asarray(new_ht_list))
            mean_val = np.mean(np.asarray(new_ht_list))
            new_list.append(medi_val)
            new_list.append(mean_val)
            clustered_lines.append(new_list)

        return clustered_lines

    @staticmethod
    def line_coeff(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float, float]:
        """
        Calculate line coefficients for the equation Ax + By + C = 0.
        From lineClassification.py lines 239-249.

        """
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]
        return a, b, -c

    @staticmethod
    def intersection(
        l1: tuple[float, float, float] | np.ndarray | list, l2: tuple[float, float, float] | np.ndarray | list
    ) -> tuple[float, float] | bool:
        """Find intersection point of two lines defined by their coefficients."""
        d = l1[0] * l2[1] - l1[1] * l2[0]
        dx = l1[2] * l2[1] - l1[1] * l2[2]
        dy = l1[0] * l2[2] - l1[2] * l2[0]
        if d != 0:
            x = dx / d
            y = dy / d
            return x, y
        return False
