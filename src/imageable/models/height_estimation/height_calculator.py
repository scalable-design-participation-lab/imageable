"""
Complete implementation of HeightCalculator.calculate_heights()
This is the main orchestration function that ties everything together
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os
import copy
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .single_view_metrology import SingleViewMetrology
from .vanishing_point import VanishingPointCalculator
from .processors.line_classifier import LineClassifier
from .processors.line_refiner import LineRefiner
from ...io.file_handler import FileHandler
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

PLTOPTS = {"color": "#33FFFF", "s": 15, "edgecolors": "none", "zorder": 5}
cmap = plt.get_cmap("jet")
norm = mcolors.Normalize(vmin=0.9, vmax=1.0)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
colors_tables = ['blue','orange','green','red','purple','brown','pink', 'gray', 'olive','cyan']

def c(x):
    return sm.to_rgba(x)


class HeightCalculator:
    """Main height calculation orchestrator - implements heightCalc from heightMeasurement.py"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.svm = SingleViewMetrology()
        self.vp_calc = VanishingPointCalculator()
        self.line_classifier = LineClassifier()
        self.line_refiner = LineRefiner()
        self.file_handler = FileHandler()
    
    def calculate_heights(
        self,
        fname_dict: Dict[str, str],
        intrins: np.ndarray,
        img_size: Optional[List[int]] = None,
        pitch: Optional[float] = None,
        use_pitch_only: int = 0,
        use_detected_vpt_only: int = 0,
        verbose: bool = False
    ) -> Optional[Dict]:
        """
        Main height calculation function - implements heightCalc from heightMeasurement.py
        
        Args:
            fname_dict: Dictionary with file paths for vpt, img, line, seg, zgt
            intrins: Camera intrinsic matrix
            img_size: Image size [width, height]
            pitch: Pitch angle in degrees
            use_pitch_only: If 1, use only pitch for VP calculation
            use_detected_vpt_only: If 1, use only detected VPs
            verbose: Enable visualization
            
        Returns:
            Dictionary with height results or None if processing fails
        """
        
        if img_size is None:
            img_size = [640, 640]
        
        try:
            # Extract filenames from dictionary
            vpt_fname = fname_dict["vpt"]
            img_fname = fname_dict["img"]
            line_fname = fname_dict["line"]
            seg_fname = fname_dict["seg"]
            zgt_fname = fname_dict.get("zgt", "none")
            
            # ######### Get vanishing points
            w, h = img_size[0], img_size[1]
            focal_length = intrins[0, 0]
            
            if use_pitch_only:
                # Initialize VPs array to match detected format
                vps = np.zeros([3, 2])
                
                # Calculate VP from pitch
                vertical_v, vline = self.vp_calc.vp_calculation_with_pitch(w, h, pitch, focal_length)
                
                # Handle special case
                if vertical_v[2] == 0:
                    vertical_v[0] = 320
                    vertical_v[1] = -9999999
                vps[2, :] = vertical_v[:2]
                
            elif '.npz' in vpt_fname and os.path.exists(vpt_fname):
                vps = self.file_handler.load_vps_2d(vpt_fname)
                
                # Optionally override with pitch-calculated VP
                if not use_detected_vpt_only and pitch is not None:
                    vertical_v, vline = self.vp_calc.vp_calculation_with_pitch(w, h, pitch, focal_length)
                    
                    if vertical_v[2] == 0:
                        vertical_v[0] = 320
                        vertical_v[1] = -9999999
                    vps[2, :] = vertical_v[:2]
            else:
                print(f"Warning: VP file not found or invalid: {vpt_fname}")
                return None
            
            # ######### Load line segments and segmentation
            line_segs, scores = self.file_handler.load_line_array(line_fname)
            seg_img = self.file_handler.load_seg_array(seg_fname)
            
            # Visualization of input data (optional)
            if verbose:
                self._visualize_inputs(img_fname, line_segs, scores, seg_img, vps, bool(use_pitch_only), vertical_v if use_pitch_only else None, vline if use_pitch_only else None, w)
            
            # ######### Process line segments
            # Filter and classify lines
            verticals = self._filter_lines_in_buildings(
                img_fname, line_segs, scores, seg_img, vps, use_pitch_only
            )
            
            # Extend vertical lines to building boundaries
            verticals = self._extend_vertical_lines(
                img_fname, verticals, seg_img, np.array([vps[2, 1], vps[2, 0]]), verbose
            )
            
            # ######### Calculate heights for each vertical line
            invK = np.linalg.inv(intrins)
            ht_set = []
            check_list = []
            
            for line in verticals:
                a = line[0]  # Top point
                b = line[1]  # Bottom point
                
                # Skip duplicate lines
                if self._is_duplicate(a, b, check_list):
                    continue
                check_list.append([a[0], a[1], b[0], b[1]])
                
                # Convert to homogeneous coordinates (swap x,y)
                a_d3 = np.asarray([a[1], a[0], 1])
                a_d3 = np.matmul(invK, np.transpose(a_d3))
                
                b_d3 = np.asarray([b[1], b[0], 1])
                b_d3 = np.matmul(invK, np.transpose(b_d3))
                
                # Calculate height based on available VP information
                if use_detected_vpt_only:
                    # Use three detected VPs
                    vps0 = np.asarray([vps[0, 0], vps[0, 1], 1])
                    vps1 = np.asarray([vps[1, 0], vps[1, 1], 1])
                    vps2 = np.asarray([vps[2, 0], vps[2, 1], 1])
                    
                    vps0 = np.matmul(invK, np.transpose(vps0))
                    vps1 = np.matmul(invK, np.transpose(vps1))
                    vps2 = np.matmul(invK, np.transpose(vps2))
                    
                    ht = self.svm.sv_measurement(
                        vps0, vps1, vps2, b_d3, a_d3,
                        zc=float(self.config["STREET_VIEW"]["CameraHeight"])
                    )
                else:
                    # Use cross-ratio with vanishing line
                    ht = self.svm.calculate_with_cross_ratio_vl(
                        vline, vertical_v[:2],
                        np.asarray([a[1], a[0]]),
                        np.asarray([b[1], b[0]]),
                        zc=float(self.config["STREET_VIEW"]["CameraHeight"])
                    )
                
                # Load ground truth if available
                ht_gt_org, ht_gt_expd = 0, 0
                if int(self.config.get("GROUND_TRUTH", {}).get("Exist", 0)) and zgt_fname != "none":
                    if os.path.exists(zgt_fname):
                        zgt_img = self.file_handler.load_zgts(zgt_fname)
                        ht_gt_org, ht_gt_expd = self._measure_ground_truth(
                            zgt_img, np.asarray([a[1], a[0]]), np.asarray([b[1], b[0]])
                        )
                
                ht_set.append([ht, a, b, ht_gt_org, ht_gt_expd])
            
            # ######### Cluster lines by height
            grouped_lines = self.line_classifier.cluster_lines_with_centers(
                ht_set, self.config, using_height=True
            )
            
            if grouped_lines is None:
                print(f'No suitable vertical lines found in image {img_fname}')
                return None
            
            # Visualize results
            if verbose:
                self._visualize_results(img_fname, seg_img, grouped_lines)
            
            # Prepare output
            heights = []
            for group in grouped_lines:
                # Last two elements are median and mean heights
                median_height = group[-2]
                mean_height = group[-1]
                heights.append({
                    'median': median_height,
                    'mean': mean_height,
                    'lines': group[:-2]  # All the line data
                })
            
            print(f"Processed: {img_fname}")
            
            return {
                'image_path': img_fname,
                'heights': heights,
                'num_buildings': len(grouped_lines),
                'vanishing_points': vps.tolist()
            }
            
        except Exception as e:
            print(f"Error processing {fname_dict.get('img', 'unknown')}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def _filter_lines_in_buildings(
        self,
        imgfile: str,
        lines: np.ndarray,
        line_scores: np.ndarray,
        segimg: np.ndarray,
        vpts: np.ndarray,
        use_vertical_vpt_only: int = 0
    ) -> List:
        """
        Filter and classify line segments (from filter_lines_outof_building_ade20k)
        """
        vert_lines = []
        t_score = float(self.config["LINE_CLASSIFY"]["LineScore"])
        
        for (a, b), s in zip(lines, line_scores):
            if s < t_score:
                continue
            
            # Check if line is in building area
            if not self.line_classifier.check_if_line_lies_in_building_area(segimg, a, b, self.config):
                continue
            
            # Classify as vertical
            if use_vertical_vpt_only or True:  # Simplified for height measurement
                is_vert = self.line_classifier.classify_with_vpts(a, b, vpts[2], self.config)
                if is_vert:
                    vert_lines.append([a, b])
        
        # Refine and merge vertical lines
        vert_line_refine = []
        for line in vert_lines:
            refined = self.line_refiner.refine_with_vpt(
                [line[0], line[1]], 
                np.asarray([vpts[2, 1], vpts[2, 0]])
            )
            vert_line_refine.append(refined)
        
        # Merge nearby lines
        vert_line_merge = self._merge_lines(vert_line_refine)
        
        return vert_line_merge
    
    def _extend_vertical_lines(
        self,
        img_name: str,
        vertical_lines: List,
        segimg: np.ndarray,
        vptz: np.ndarray,
        verbose: bool = False
    ) -> List:
        """
        Extend vertical lines to building boundaries (from verticalLineExtending)
        """
        extd_lines = []
        
        for line in vertical_lines:
            # Refine with VP
            line = self.line_refiner.refine_with_vpt(line, vptz)
            a = line[0]
            b = line[1]
            
            # Extend to building boundaries
            extd_a, extd_b = self.line_refiner.extend_lines(a, b, segimg, self.config)
            
            if len(extd_a) == 0 or len(extd_b) == 0:
                continue
                
            extd_lines.append([extd_a, extd_b])
        
        return extd_lines
    
    def _merge_lines(self, lines: List) -> List:
        """Merge nearby parallel lines"""
        merged = []
        used = [False] * len(lines)
        
        for i in range(len(lines)):
            if used[i]:
                continue
                
            line_i = lines[i]
            # Check length
            if np.linalg.norm(line_i[0] - line_i[1]) < 10:
                continue
            
            # Try to merge with other lines
            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue
                    
                line_j = lines[j]
                # Simple distance check
                mid_i = (line_i[0] + line_i[1]) / 2
                mid_j = (line_j[0] + line_j[1]) / 2
                
                if np.linalg.norm(mid_i - mid_j) < 5:  # Threshold
                    # Merge: take extremes
                    if line_i[0][0] > line_j[0][0]:  # Compare y-coordinates
                        line_i[0] = line_j[0]
                    if line_i[1][0] < line_j[1][0]:
                        line_i[1] = line_j[1]
                    used[j] = True
            
            merged.append(line_i)
            used[i] = True
        
        return merged
    
    def _is_duplicate(self, a: np.ndarray, b: np.ndarray, check_list: List) -> bool:
        """Check if line (a,b) is duplicate"""
        for a0, a1, b0, b1 in check_list:
            if a0 == a[0] and a1 == a[1] and b0 == b[0] and b1 == b[1]:
                return True
        return False
    
    def _measure_ground_truth(
        self,
        zgt_img: np.ndarray,
        a: np.ndarray,
        b: np.ndarray
    ) -> Tuple[float, float]:
        """Measure ground truth height if available"""
        if a[1] > b[1]:
            a, b = b, a
        
        # Convert to integer coordinates - FIX np.cast syntax
        a = np.array(a + [0.5, 0.5], dtype=int)
        b = np.array(b + [0.5, 0.5], dtype=int)
        
        rows, cols = zgt_img.shape
        
        # Boundary checks
        a[0] = min(cols - 1, max(0, a[0]))
        a[1] = min(rows - 1, max(0, a[1]))
        b[0] = min(cols - 1, max(0, b[0]))
        b[1] = min(rows - 1, max(0, b[1]))
        
        # Calculate ground truth
        if zgt_img[a[1], a[0]] == 0 or zgt_img[b[1], b[0]] == 0:
            gt_org = 0
        else:
            gt_org = abs(zgt_img[a[1], a[0]] - zgt_img[b[1], b[0]])
        
        # Extended calculation (simplified)
        gt_expd = gt_org  # Simplified for now
        
        return float(gt_org), float(gt_expd)
    
    def _visualize_inputs(
        self,
        img_fname: str,
        line_segs: np.ndarray,
        scores: np.ndarray,
        seg_img: np.ndarray,
        vps: np.ndarray,
        use_pitch_only: bool,
        vertical_v: Optional[np.ndarray],
        vline: Optional[np.ndarray],
        w: int
    ):
        """Visualize input data"""
        org_image = skimage.io.imread(img_fname)
        
        plt.figure(figsize=(10, 8))
        plt.gca().set_axis_off()
        plt.imshow(org_image)
        plt.imshow(seg_img, alpha=0.5)
        
        # Draw lines
        t = 0.94  # Score threshold
        for (a, b), s in zip(line_segs, scores):
            if s < t:
                continue
            plt.plot([a[1], b[1]], [a[0], b[0]], c=c(s), linewidth=2, zorder=s)
            plt.scatter(a[1], a[0])
            plt.scatter(b[1], b[0])
        
        # Show vanishing points
        if use_pitch_only and vertical_v is not None and vline is not None:
            x, y = vertical_v[:2]
            plt.scatter(x, y, c='red', s=100, marker='*')
            plt.plot([0, w], [vline[2], vline[2]], c='b', linewidth=3)
        else:
            for i in range(len(vps)):
                x, y = vps[i]
                plt.scatter(x, y, c='red', s=100, marker='*')
        
        plt.title("Input: Lines, Segmentation, and Vanishing Points")
        plt.show()
    
    def _visualize_results(
        self,
        img_fname: str,
        seg_img: np.ndarray,
        grouped_lines: List
    ):
        """Visualize final height results"""
        org_img = skimage.io.imread(img_fname)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(org_img)
        plt.imshow(seg_img, alpha=0.3)
        
        heights = []
        ax_legends = []
        
        for i, group in enumerate(grouped_lines):
            # Extract heights (last two elements are statistics)
            median_height = group[-2]
            mean_height = group[-1]
            heights.append([median_height, mean_height])
            
            # Draw lines in this group
            for j in range(len(group) - 2):
                ht, a, b, *_ = group[j]
                color = colors_tables[i % len(colors_tables)]
                ax_line, = plt.plot([a[1], b[1]], [a[0], b[0]], c=color, linewidth=3)
                plt.scatter(a[1], a[0], **PLTOPTS)
                plt.scatter(b[1], b[0], **PLTOPTS)
            
            if len(group) > 2:  # Only add legend if there are lines
                ax_legends.append(ax_line)
        
        # Add legend with height information
        if ax_legends:
            plt.legend(ax_legends, 
                      [f'Building {i+1}: mean={h[1]:.2f}m, median={h[0]:.2f}m' 
                       for i, h in enumerate(heights)])
        
        plt.title("Height Estimation Results")
        
        # Save if output directory exists
        result_save_name = img_fname.replace('imgs', 'ht_results').replace('.jpg', '_result.png')
        result_dir = os.path.dirname(result_save_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir, exist_ok=True)
        plt.savefig(result_save_name, bbox_inches="tight", dpi=100)
        
        plt.show()
        plt.close()

