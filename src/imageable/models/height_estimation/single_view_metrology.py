import numpy as np
from typing import Tuple
from numpy.typing import NDArray

class SingleViewMetrology:
    """Pure mathematical functions for height calculation"""
    
    @staticmethod
    def sv_measurement(v1, v2, v3, x1, x2, zc=2.5):
        vline = np.cross(v1, v2)
        p4 = vline / np.linalg.norm(vline)
        
        zc = zc * np.linalg.det([v1, v2, v3])
        alpha = -np.linalg.det([v1, v2, p4]) / zc
        p3 = alpha * v3
        
        zx = -np.linalg.norm(np.cross(x1, x2)) / (np.dot(p4, x1) * np.linalg.norm(np.cross(p3, x2)))
        zx = abs(zx)
        
        return zx
    
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
        pass