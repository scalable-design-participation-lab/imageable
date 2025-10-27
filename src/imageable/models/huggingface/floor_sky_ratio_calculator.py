import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from imageable.models.huggingface.base import HuggingFaceModelWrapper
from imageable.utils.masks.mask_operations import get_mask_centroid, get_mask_limits


class FloorSkyRatioCalculator(HuggingFaceModelWrapper):
    """
    Calculates the ratio of floor pixels and sky pixels below and
    above a building, respectively. These quantities are used to
    adjust the camera position, so that the complete building can be observed.
    """

    def __init__(
        self, model_repo: str = "urilp4669/Facade_Segmentator", filename: str = "facades.pt", device: str | None = None
    ) -> None:
        self.model_repo = model_repo
        self.filename = filename
        self.device = device or self._resolve_device()
        self.model = None

    def load_model(self) -> None:
        """Load the YOLO model from the Hugging Face repository."""
        model_path = hf_hub_download(repo_id=self.model_repo, filename=self.filename)
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None

    def preprocess(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Convert to numpy array if PIL Image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        return image

    def postprocess(
        self, outputs: Results, sky_label: str = "sky", floor_label: str = "sidewalk", building_label: str = "facade"
    ) -> dict:
        """
        Compute sky/floor pixel ratios and return their binary masks. In the case that
        a building facade is detected near the center of the image, the ratios will focus
        on the areas above and below the facade.

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
        # Get the masks and names dictionary
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

        facade_masks = []
        # Get whole binary masks
        for mask, name in zip(mask_array, class_names, strict=False):
            binary = mask > 0.5
            if name == sky_label:
                sky_mask |= binary
            elif name == floor_label:
                floor_mask |= binary
            elif name == building_label:
                facade_masks.append(binary)

        # Before obtaining the ratios we are going to multiply
        # if possible by a mask of the building closest to the
        # center of the image.
        closest_facade = None
        distance_to_center = float("inf")
        if len(facade_masks) > 0:
            for i in range(len(facade_masks)):
                centroid = get_mask_centroid(facade_masks[i])
                centroid_x = centroid[0]
                diff_x = np.abs(centroid_x - width / 2)
                if diff_x < distance_to_center:
                    distance_to_center = diff_x
                    closest_facade = facade_masks[i]

        # If closest_facade is None we simply return the sky_ratio and floor_ratios
        # without multiplying
        if closest_facade is None:
            total_pixels = height * width
            sky_ratio = sky_mask.sum() / total_pixels
            floor_ratio = floor_mask.sum() / total_pixels

            return {"sky_ratio": sky_ratio, "floor_ratio": floor_ratio, "sky_mask": sky_mask, "floor_mask": floor_mask}

        # Get limits of the facade
        facade_bounds = get_mask_limits(closest_facade)
        x_min = facade_bounds[0]
        x_max = facade_bounds[2]

        # Ok facade tunnel mask (i am going to call it that)
        # should be easy to obtain.
        tube_mask = np.array([[1 if x_min <= i <= x_max else 0 for i in range(width)] for j in range(height)])

        # limit the sky and floor to this tube
        limited_sky = sky_mask & tube_mask
        limited_floor = floor_mask & tube_mask

        sky_ratio = limited_sky.sum() / tube_mask.sum()
        floor_ratio = limited_floor.sum() / tube_mask.sum()

        return {
            "sky_ratio": sky_ratio,
            "floor_ratio": floor_ratio,
            "sky_mask": sky_mask,
            "floor_mask": floor_mask,
            "limited_sky": limited_sky,
            "limited_floor": limited_floor,
            "facade_mask": closest_facade,
        }

    def predict(self, image: Image.Image | np.ndarray, conf: float = 0.5) -> dict:
        """
        Predict the sky and floor ratios for a given image.

        Parameters
        ----------
        image
            The input image for which to predict the sky and floor ratios.
        conf
            The confidence threshold for the detection of sky and floor
            on the image.

        Returns
        -------
        results_dictionary
            A dictionary containing the predicted sky and floor ratios,
            as well as the corresponding masks.
        """
        if not self.is_loaded():
            self.load_model()
        input_tensor = self.preprocess(image)
        results = self.model.predict(input_tensor, conf=conf, verbose=False)
        return self.postprocess(results[0])

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
