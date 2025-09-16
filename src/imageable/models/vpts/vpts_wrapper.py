from imageable.models.base import BaseModelWrapper
import numpy as np
from typing import Any
from lu_vp_detect import VPDetection

class VPTSWrapper(BaseModelWrapper):
    
    def __init__(self)->None:
        """Initializes the VPTSWrapper class."""
        super().__init__()
        self.model_name = "vpts"
        self.load_model()

    def load_model(self)->None:
        """In this simple version there is no model to load."""
        pass

    def is_loaded(self)->bool:
        """Always returns True as there is no model to load."""
        return True

    def preprocess(self, image:np.ndarray)->np.ndarray:
        """No preprocessing needed, return the image as is."""
        return image
    
    def postprocess(self, outputs:dict[str, Any])->dict[str, Any]:
        vpts_2d = outputs["vpts_2d"]
        K = outputs["K"]
        
        # We will first identify the vertical vanishing point
        #It's -1 because the y axis is inverted in images
        up_direction = np.array([0, -1, 0])
        
        #Project the 2d points to 3d
        vector_dirs = []
        K_inv = np.linalg.inv(K)
        for vp_2d in vpts_2d:
            vp_homog = np.array([vp_2d[0], vp_2d[1], 1.0])
            dir_3d = K_inv @ vp_homog
            dir_3d = dir_3d / np.linalg.norm(dir_3d)
            vector_dirs.append(dir_3d)
        
        #Let's get scores to identify the vertical vanishing point
        scores = []
        for dir_3d in vector_dirs:
            score = np.abs(np.dot(dir_3d, up_direction))
            scores.append(score)
        
        max_index = np.argmax(scores)
        vpt_vertical_2d = vpts_2d[max_index]
        vpt_vertical_3d = outputs["vpts_3d"][max_index]
        
        #Now we order the other two vanishing points
        other_indices = [i for i in range(3) if i != max_index]
        sorted_other_indices = sorted(
            other_indices, 
            key=lambda i: vpts_2d[i][0])
        
        final_indices = sorted_other_indices + [max_index]
        vpts_2d_ordered = vpts_2d[final_indices]
        vpts_3d_ordered = outputs["vpts_3d"][final_indices]
        
        return{
            "vpts_3d": vpts_3d_ordered,
            "vpts_2d": vpts_2d_ordered,
            "K": K
        }
        
            
        
        

    def predict(
        self,
        image:np.ndarray,
        FOV:float = 90.0, 
        seed:int = None,
        length_threshold:float = 60
        
    )->dict[str, Any]:
        """
        Obtains the vanishing points in 3d and 2d using the lu-vp method.
        Parameters
        __________
        image
            The input image as a numpy array.
        FOV
            Field of view used to obtain the image. Default is 90 degrees.
        Returns
        _______
        vpts_dict
            A dictionary structured as follows:
                {
                    "vpts_3d": np.ndarray of shape (3, 3) with the 3D vanishing points,
                    "vpts_2d": np.ndarray of shape (3, 2) with the 2D vanishing points,
                    "K": An estimation of the camera instrinsics matrix.
                }
        """
        W = np.size(image, 1)
        H = np.size(image, 0)
        cx = W / 2
        cy = H / 2
        principal_point = (cx, cy)
        f = W / (2 * np.tan(FOV * np.pi / 360))
        
        # Let's obtain the points
        vp_detector = VPDetection(
            length_threshold,
            principal_point,
            f,
            seed
        )
        
        vp_detector.find_vps(image)
        vpts_3d = np.array(vp_detector.vps)
        vpts_2d = np.array(vp_detector.vps_2D)
        
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        
        vpts_dict = {
            "vpts_3d": vpts_3d,
            "vpts_2d": vpts_2d,
            "K": K
        }
        
        return self.postprocess(vpts_dict)
        
        
    
    