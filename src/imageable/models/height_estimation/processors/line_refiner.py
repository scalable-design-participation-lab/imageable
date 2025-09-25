import numpy as np
import copy

class LineRefiner:
    
    def extend_lines(self, pt1, pt2, segmt, config):
        """From lineRefinement.py lines 9-92"""
        sky_label = config["SEGMENTATION"]["SkyLabel"].split(",")
        building_label = int(config["SEGMENTATION"]["BuildingLabel"])
        # Fix np.cast syntax:
        ground_label = np.array(config["SEGMENTATION"]["GroundLabel"].split(','), dtype=int)
        edge_thres = np.array(config["LINE_REFINE"]["Edge_Thres"].split(','), dtype=int)
        
        if pt1[0] > pt2[0]:
            pt_up = pt2
            pt_down = pt1
        else:
            pt_up = pt1
            pt_down = pt2
        
        if np.linalg.norm(pt_down - pt_up) == 0:
            return [], []
        
        direction = (pt_down - pt_up) / np.linalg.norm(pt_down - pt_up)
        pt_up_end = copy.deepcopy(pt_up)
        pt_down_end = copy.deepcopy(pt_down)
        pt_middle = (pt_up + pt_down) / 2.0
        
        rows, cols = segmt.shape
        
        # Boundary checks
        if pt_up_end[0] > rows - 2:
            pt_up_end[0] = rows - 2
        if pt_up_end[1] > cols - 2:
            pt_up_end[1] = cols - 2
        if pt_down_end[0] > rows - 2:
            pt_down_end[0] = rows - 2
        if pt_down_end[1] > cols - 2:
            pt_down_end[1] = cols - 2
        
        if pt_middle[0] >= rows - 1 or pt_middle[1] >= cols - 1:
            return [], []
        
        # Check if points are in building
        # Fix np.cast syntax:
        if (segmt[int(pt_up_end[0] + 0.5)][int(pt_up_end[1] + 0.5)] != building_label or
            segmt[int(pt_down_end[0] + 0.5)][int(pt_down_end[1] + 0.5)] != building_label or
            segmt[int(pt_middle[0] + 0.5)][int(pt_middle[1] + 0.5)] != building_label):
            return [], []
        
        # Extend upward until sky
        flag = 1
        while flag:
            pt_up_end = pt_up_end - direction
            if pt_up_end[0] < 0 or pt_up_end[1] < 0 or pt_up_end[1] >= rows - 1:
                flag = 0
                pt_up_end = pt_up_end + direction
                continue
            if segmt[int(pt_up_end[0] + 0.5)][int(pt_up_end[1] + 0.5)] == int(sky_label[0]) or segmt[int(pt_up_end[0] + 0.5)][int(pt_up_end[1] + 0.5)] == int(sky_label[1]):
                flag = 0
                pt_up_end = pt_up_end + direction
        
        # Extend downward until ground
        flag = 1
        out_of_building = False
        while flag:
            pt_down_end = pt_down_end + direction
            if pt_down_end[0] >= cols - 1 or pt_down_end[1] < 0 or pt_down_end[1] >= rows - 1:
                flag = 0
                continue
            
            current_label = segmt[int(pt_down_end[0] + 0.5)][int(pt_down_end[1] + 0.5)]
            
            if current_label != building_label and current_label not in ground_label:
                out_of_building = True
            else:
                if current_label == building_label:
                    out_of_building = False
            
            if current_label in ground_label and not out_of_building:
                flag = 0
                pt_down_end = pt_down_end - direction
            else:
                if current_label in ground_label:
                    return [], []
        
        # Check edge thresholds
        if (pt_up_end[0] > cols - 1 - edge_thres[0] or
            pt_up_end[1] < edge_thres[0] or pt_up_end[1] > rows - edge_thres[0] or
            pt_down_end[0] < edge_thres[0] or pt_down_end[0] > cols - 1 - edge_thres[0] or
            pt_down_end[1] < edge_thres[0] or pt_down_end[1] > rows - 1 - edge_thres[0]):
            return [], []
        
        return pt_up_end, pt_down_end
    
    @staticmethod
    def point_on_line(a, b, p):
        """From lineRefinement.py lines 235-261"""
        l2 = np.sum((a - b) ** 2)
        if l2 == 0:
            print('p1 and p2 are the same points')
            return p
        
        t = np.sum((p - a) * (b - a)) / l2
        projection = a + t * (b - a)
        return projection
    
    def refine_with_vpt(self, line, vpt):
        """From lineRefinement.py lines 264-279"""
        a = line[0]
        b = line[1]
        mpt = (a + b) / 2.0
        line[0] = self.point_on_line(vpt, mpt, a)
        line[1] = self.point_on_line(vpt, mpt, b)
        return line