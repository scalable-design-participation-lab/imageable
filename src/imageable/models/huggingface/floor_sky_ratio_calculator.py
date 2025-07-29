from imageable.models.huggingface.base import HuggingFaceModelWrapper
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import torch

class FloorSkyRatioCalculator(HuggingFaceModelWrapper):
    def __init__(self, 
        model_repo: str = "urilp4669/Facade_Segmentator", 
        filename: str = "facades.pt", 
        device: str | None = None) -> None:

        self.model_repo = model_repo
        self.filename = filename
        self.device = device or self._resolve_device()
        self.model = None

    def load_model(self):
        model_path = hf_hub_download(repo_id=self.model_repo, filename=self.filename)
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
        return image
    


    def postprocess(self, outputs, sky_label="sky", floor_label="sidewalk") -> dict:
        """
        Computes sky/floor pixel ratios and returns their binary masks.

        Parameters
        ----------
        outputs : ultralytics.engine.results.Results
            The YOLO model output.
        sky_label : str
            Class name for sky.
        floor_label : str
            Class name for sidewalk/floor.

        Returns
        -------
        dict
            {
                "sky_ratio": float,
                "floor_ratio": float,
                "sky_mask": np.ndarray[bool],
                "floor_mask": np.ndarray[bool]
            }
        """

        #Get the masks and names dictionary
        masks = outputs.masks
        names = outputs.names
        if masks is None:
            return {
                "sky_ratio": 0.0,
                "floor_ratio": 0.0,
                "sky_mask": None,
                "floor_mask": None,
            }

        class_ids = outputs.boxes.cls.int().tolist()
        class_names = [names[cid] for cid in class_ids]
        mask_array = masks.data.cpu().numpy()  

        height, width = masks.data.shape[1], masks.data.shape[2]
        sky_mask = np.zeros((height, width), dtype=bool)
        floor_mask = np.zeros((height, width), dtype=bool)
        
        #Get whole binary masks
        for mask, name in zip(mask_array, class_names):
            binary = mask > 0.5
            if name == sky_label:
                sky_mask |= binary
            elif name == floor_label:
                floor_mask |= binary

        # Obtain the ratios
        total_pixels = height * width
        sky_ratio = sky_mask.sum() / total_pixels
        floor_ratio = floor_mask.sum() / total_pixels

        return {
            "sky_ratio": sky_ratio,
            "floor_ratio": floor_ratio,
            "sky_mask": sky_mask,
            "floor_mask": floor_mask,
        }

    def predict(self, image: Image.Image | np.ndarray, conf = 0.5):
        if not self.is_loaded():
            self.load_model()
        input_tensor = self.preprocess(image)
        results = self.model.predict(image, conf=conf)
        return self.postprocess(results[0])

    def _resolve_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    

