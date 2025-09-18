import numpy as np
from typing import Dict, List, Optional
from .single_view_metrology import SingleViewMetrology
from .vanishing_point import VanishingPointCalculator
from .processors.line_classifier import LineClassifier
from .processors.line_refiner import LineRefiner

class HeightCalculator:
    
    def __init__(self, config: Dict):
        self.config = config
        self.svm = SingleViewMetrology()
        self.vp_calc = VanishingPointCalculator()
        self.line_classifier = LineClassifier()
        self.line_refiner = LineRefiner()
    
    def calculate_heights(self, fname_dict, intrins, img_size=None, 
                          pitch=None, use_pitch_only=0, 
                          use_detected_vpt_only=0, verbose=False):
        pass