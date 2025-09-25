import copy
import numpy as np
from sklearn.cluster import DBSCAN

class LineClassifier:
    
    def classify_with_vpts(self, n1, n2, vpt, config):
        """From lineClassification.py lines 15-51"""
        flag = False
        t_angle = float(config["LINE_CLASSIFY"]["AngleThres"])
        
        p1 = np.array([n1[1], n1[0]])
        p2 = np.array([n2[1], n2[0]])
        
        mpt = [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
        d1 = p2 - p1
        d2 = vpt - mpt
        angle = np.rad2deg(np.arccos(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))))
        
        if angle < t_angle or 180 - angle < t_angle:
            flag = True
        
        return flag
    
    def check_if_line_lies_in_building_area(self, seg_img, a, b, config):
        """From lineClassification.py lines 54-104"""
        middle = (a + b)/2.0
        norm_direction = (a - b) / np.linalg.norm(a - b)
        ppd_dir = np.asarray([norm_direction[1], -norm_direction[0]])
        
        building_label = int(config["SEGMENTATION"]["BuildingLabel"])
        # Fix np.cast syntax:
        ground_label = np.array(config["SEGMENTATION"]["GroundLabel"].split(','), dtype=int)
        
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
        print(point_check_list)
        
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
            else:
                if np.remainder(total_num, 3) == 0:
                    local_num = 0
        return flag
    
    def cluster_lines_with_centers(self, ht_set, config, using_height=False):
        """From lineClassification.py lines 436-478"""
        X = []
        if using_height:
            for ht, a, b, *_ in ht_set:
                X.append([(a[0] + b[0])/2, (a[1] + b[1])/2, ht])
        else:
            for ht, a, b, *_ in ht_set:
                X.append([(a[0] + b[0])/2, (a[1] + b[1])/2])
        X = np.asarray(X)
        
        max_DBSAN_dist = float(config["HEIGHT_MEAS"]["MaxDBSANDist"])
        
        try:
            clustering = DBSCAN(eps=max_DBSAN_dist, min_samples=1).fit(X)
        except:
            print("Error in clustering")
            return None
        
        clustered_lines = []
        max_val = np.max(clustering.labels_)+1
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
    def line_coeff(p1, p2):
        """Helper function from lineClassification.py lines 239-249"""
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C
    
    @staticmethod
    def intersection(L1, L2):
        """Helper function from lineClassification.py lines 252-267"""
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return False