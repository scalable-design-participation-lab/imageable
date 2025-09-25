import numpy as np
from pathlib import Path

class FileHandler:
    """Unified file I/O operations from filesIO.py"""
    
    @staticmethod
    def load_vps_2d(filename: str):
        """Load vanishing points"""
        with np.load(filename) as npz:
            vpts_pd_2d = npz['vpts_re']
        return vpts_pd_2d
    
    @staticmethod
    def load_line_array(filename: str):
        """Load lines and scores"""
        with np.load(filename) as npz:
            nlines = npz["nlines"]
            nscores = npz["nscores"]
        return nlines, nscores
    
    @staticmethod
    def load_seg_array(filename: str):
        """Load segmentation results"""
        with np.load(filename) as npz:
            seg_array = npz["seg"]
        return seg_array
    
    @staticmethod
    def load_zgts(filename: str):
        """Load ground truth height data"""
        with np.load(filename) as npz:
            zgt = npz["height"]
        return zgt