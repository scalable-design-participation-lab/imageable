from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
import numpy as np
from typing import Optional, Any
from .base import HuggingFaceModelWrapper
import json
from pathlib import Path


class SegformerSegmentationWrapper(HuggingFaceModelWrapper):
    """
    A wrapper for the SegFormer semantic segmentation model from Hugging Face.

    This class loads a pretrained SegFormer model and its associated image processor,
    performs inference on PIL images, supports label remapping, and colorizes
    segmentation maps using a predefined ADE20K palette.

    Parameters
    ----------
    model_name : str
        The Hugging Face model identifier (e.g., "nvidia/segformer-b5-finetuned-ade-640-640").
    device : Optional[str], default=None
        The device to use for inference. Automatically resolves to "cuda", "mps", or "cpu" if not specified.
    """

    def __init__(self, model_name: str, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or self._resolve_device()
        self.model = None
        self.processor = None
        self._ade_palette = self._load_palette_from_json(
            "src/imageable/assets/ade20k_palette.json"
        )

    def load_model(self):
        """
        Loads the pretrained SegFormer model and associated image processor from Hugging Face.
        """
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        # Move model to the specified devic
        self.model = self.model.to(self.device)  # type: ignore

    def is_loaded(self) -> bool:
        """
        Checks if the model and processor have been successfully loaded.

        Returns
        -------
        bool
            True if both model and processor are loaded, False otherwise.
        """
        return self.model is not None and self.processor is not None

    def predict(self, image: Image.Image) -> np.ndarray:
        """
        Runs semantic segmentation on an input image.

        Parameters
        ----------
        image : PIL.Image.Image
            The input image in RGB format.

        Returns
        -------
        np.ndarray
            A 2D array of predicted class indices (same spatial dimensions as the input image).
        """
        if not self.is_loaded():
            self.load_model()
        assert self.model is not None
        assert self.processor is not None

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits

        resized_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (width, height)
            mode="bicubic",
            align_corners=False,
        )

        prediction = torch.argmax(resized_logits, dim=1).squeeze().cpu().numpy()
        return prediction

    def remap_labels(self, seg: np.ndarray, mapping: dict) -> np.ndarray:
        """
        Remaps segmentation class indices using a provided dictionary.

        Parameters
        ----------
        seg : np.ndarray
            A 2D segmentation map of class indices.
        mapping : dict
            A mapping of source indices to target indices (e.g., {2: 1, 3: 2}).

        Returns
        -------
        np.ndarray
            The remapped segmentation array.
        """
        return self._remap_labels(seg, mapping)

    def colorize(self, seg: np.ndarray) -> Image.Image:
        """
        Applies a color palette to a segmentation map to create a colorized image.

        Parameters
        ----------
        seg : np.ndarray
            A 2D array of class indices.

        Returns
        -------
        PIL.Image.Image
            A colorized segmentation image in "P" mode.
        """
        return self._apply_palette(seg, self._ade_palette)

    # --- Private helpers ---

    def _resolve_device(self) -> str:
        """
        Detects the best available device ("cuda", "mps", or "cpu").

        Returns
        -------
        str
            The selected device identifier.
        """
        if torch.cuda.is_available():
            return "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _remap_labels(self, seg: np.ndarray, mapping: dict) -> np.ndarray:
        """
        Internal utility to remap segmentation labels.

        Parameters
        ----------
        seg : np.ndarray
            Input segmentation map.
        mapping : dict
            Dictionary mapping old labels to new labels.

        Returns
        -------
        np.ndarray
            Remapped segmentation map.
        """
        remapped = np.zeros_like(seg)
        for src, tgt in mapping.items():
            remapped[seg == src] = tgt
        return remapped

    def _apply_palette(self, seg: np.ndarray, palette: list) -> Image.Image:
        """
        Applies a custom color palette to a label image.

        Parameters
        ----------
        seg : np.ndarray
            Segmentation map of class indices.
        palette : list
            Flat list of RGB values for each class.

        Returns
        -------
        PIL.Image.Image
            Colorized image in palette mode.
        """
        img = Image.fromarray(seg.astype("uint8"), mode="P")
        img.putpalette(palette)
        return img

    def _load_palette_from_json(self, path: str) -> list[int]:
        """
        Loads the ADE20K color palette from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON file containing the flat list of RGB values.

        Returns
        -------
        list[int]
            The loaded RGB color palette.
        """
        with open(Path(path), "r", encoding="utf-8") as f:
            return json.load(f)
