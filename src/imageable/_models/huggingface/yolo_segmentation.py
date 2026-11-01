from typing import Any, ClassVar

import numpy as np
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from PIL import Image
from imageable._models.huggingface.base import HuggingFaceModelWrapper
from ultralytics import YOLO  # type: ignore[attr-defined]
from ultralytics.engine.results import Results

class YOLOSegmentationWrapper(HuggingFaceModelWrapper):
    """Segments facades with the YOLO model into the labeled image that the height estimation reads."""

    HUGGING_FACE_REPO = "walup/facades_segmentation"
    MODEL_NAME = "facades.pt"
    REMAPPING_DICT: ClassVar[dict[str, int]] = {"sky": 2, "road": 11, "sidewalk": 11, "facade": 1}
    OCCLUDING_CLASSES = ("bench", "person", "street light", "trash", "utility pole", "vegetation", "vehicle")

    def __init__(
            self,
            model_path:str | None = None
        )->None:

        self.model_path = model_path
        self.model: YOLO | None = None



    def load_model(self)->None:
        """Load the YOLO model from model_path, or download it from Hugging Face."""
        if(not self.model_path is None):
            self.model = YOLO(self.model_path)
        else:
            self.download_model()

    def download_model(self)->None:
        """Download the YOLO model from Hugging Face, unless it is already in the cache."""
        cached = try_to_load_from_cache(
            repo_id=self.HUGGING_FACE_REPO,
            filename=self.MODEL_NAME,
        )

        if cached is None:
            model_path = hf_hub_download(
                repo_id=self.HUGGING_FACE_REPO,
                filename=self.MODEL_NAME,
            )
        else:
            model_path = cached

        self.model_path = model_path
        self.model = YOLO(self.model_path)

    def predict(self, inputs:np.ndarray[Any, np.dtype[np.uint8]])->np.ndarray[Any, np.dtype[np.int_]]:
        """Return the labeled image (H, W) of an RGB image: 1 building, 2 sky, 11 ground and 0 the rest."""
        if(not self.is_loaded()):
            self.load_model()
        assert self.model is not None, "Model is not loaded."
        results = self.model.predict(self.preprocess(inputs), retina_masks=True, verbose=False)
        return self.postprocess(results[0])


    def is_loaded(self)->bool:
        """Check if the model is loaded."""
        return self.model is not None


    def preprocess(self, inputs:np.ndarray[Any, np.dtype[np.uint8]])->Image.Image:
        """Convert the RGB array to a PIL image, because YOLO reads a NumPy array as BGR."""
        return Image.fromarray(inputs)

    def postprocess(self, outputs:Results)->np.ndarray[Any, np.dtype[np.int_]]:
        """Paint the masks into one labeled image: sky and ground, then facade, then the occluding classes as 0."""
        labeled_image = np.zeros(outputs.orig_shape, dtype=np.int_)
        if(outputs.masks is None or outputs.boxes is None):
            return labeled_image
        class_names = np.array([outputs.names[int(class_id)] for class_id in outputs.boxes.cls])
        masks = outputs.masks.data.cpu().numpy().astype(bool)  # type: ignore[union-attr]
        for class_name, label in self.REMAPPING_DICT.items():
            labeled_image[masks[class_names == class_name].any(axis=0)] = label
        labeled_image[masks[np.isin(class_names, self.OCCLUDING_CLASSES)].any(axis=0)] = 0
        return labeled_image
