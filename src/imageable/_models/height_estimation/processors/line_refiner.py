import copy
from typing import Any, cast

import numpy as np


class LineRefiner:
    def extend_lines(
        self, pt1: np.ndarray, pt2: np.ndarray, segmt: np.ndarray, config: dict[str, Any]
    ) -> tuple[np.ndarray | list[Any], np.ndarray | list[Any]]:
        """
        Extend a vertical line segment to the roof and to the ground, as SIHE does.

        - Extend upward until reaching the sky.
        - Extend downward until reaching the ground.
        - Reject the line when the last label above the ground is not building.
        """
        building_label = int(config["SEGMENTATION"]["BuildingLabel"])
        sky_label = np.array(config["SEGMENTATION"]["SkyLabel"].split(","), dtype=int)
        ground_label = np.array(config["SEGMENTATION"]["GroundLabel"].split(","), dtype=int)
        edge_thres = np.array(config["LINE_REFINE"]["Edge_Thres"].split(","), dtype=int)

        rows, cols = segmt.shape

        # ensure float for safe incremental updates
        pt1 = pt1.astype(float)
        pt2 = pt2.astype(float)

        # define up/down by row index (0 = top)
        if pt1[0] <= pt2[0]:
            pt_up = pt1
            pt_down = pt2
        else:
            pt_up = pt2
            pt_down = pt1

        # degenerate line
        if np.linalg.norm(pt_down - pt_up) == 0:
            return [], []

        # direction from top to bottom
        direction = (pt_down - pt_up) / np.linalg.norm(pt_down - pt_up)

        pt_up_end = copy.deepcopy(pt_up)
        pt_down_end = copy.deepcopy(pt_down)

        # clamp initial endpoints inside image
        for p in (pt_up_end, pt_down_end):
            p[0] = np.clip(p[0], 0, rows - 1)
            p[1] = np.clip(p[1], 0, cols - 1)

        # require endpoints and middle point to start in building
        pt_middle = (pt_up_end + pt_down_end) / 2.0
        if (
            segmt[int(pt_up_end[0] + 0.5), int(pt_up_end[1] + 0.5)] != building_label
            or segmt[int(pt_down_end[0] + 0.5), int(pt_down_end[1] + 0.5)] != building_label
            or segmt[int(pt_middle[0] + 0.5), int(pt_middle[1] + 0.5)] != building_label
        ):
            return [], []

        # ---- extend upward until reaching the sky ----
        pt_cur = pt_up_end.copy()
        while True:
            r = int(pt_cur[0] + 0.5)
            c = int(pt_cur[1] + 0.5)

            if r < 0 or r >= rows or c < 0 or c >= cols:
                break

            if segmt[r, c] in sky_label:
                # last point below the sky is pt_up_end
                break

            pt_up_end = pt_cur.copy()
            pt_cur = pt_cur - direction

        # ---- extend downward until reaching the ground ----
        pt_cur = pt_down_end.copy()
        out_of_building = False
        while True:
            r = int(pt_cur[0] + 0.5)
            c = int(pt_cur[1] + 0.5)

            if r < 0 or r >= rows or c < 0 or c >= cols:
                break

            if segmt[r, c] in ground_label:
                if out_of_building:
                    # something that is not building hides the base
                    return [], []
                # last point above the ground is pt_down_end
                break

            out_of_building = segmt[r, c] != building_label
            pt_down_end = pt_cur.copy()
            pt_cur = pt_cur + direction

        # if extension collapsed to nothing (should be rare)
        if np.linalg.norm(pt_down_end - pt_up_end) == 0:
            return [], []

        # ---- edge threshold check (rows vs cols fixed) ----
        # edge_thres[0] -> vertical margin, edge_thres[1] -> horizontal margin
        v_margin = edge_thres[0]
        h_margin = edge_thres[1] if edge_thres.size > 1 else edge_thres[0]

        def is_near_edge(p: np.ndarray) -> bool:
            r = p[0]
            c = p[1]
            return bool(
                r < v_margin
                or r > rows - 1 - v_margin
                or c < h_margin
                or c > cols - 1 - h_margin
            )

        if is_near_edge(pt_up_end) or is_near_edge(pt_down_end):
            return [], []

        return pt_up_end, pt_down_end

    @staticmethod
    def point_on_line(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
        """From lineRefinement.py lines 235-261"""
        l2 = np.sum((a - b) ** 2)
        if l2 == 0:
            print("p1 and p2 are the same points")
            return p

        t = np.sum((p - a) * (b - a)) / l2
        projection = a + t * (b - a)
        return cast("np.ndarray", projection)

    def refine_with_vpt(self, line: list[np.ndarray], vpt: np.ndarray) -> list[np.ndarray]:
        """From lineRefinement.py lines 264-279"""
        a = line[0]
        b = line[1]
        mpt = (a + b) / 2.0
        line[0] = self.point_on_line(vpt, mpt, a)
        line[1] = self.point_on_line(vpt, mpt, b)
        return line
