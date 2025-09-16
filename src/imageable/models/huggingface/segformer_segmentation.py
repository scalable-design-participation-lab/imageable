import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

from imageable.models.huggingface.base import HuggingFaceModelWrapper


class SegformerSegmentationWrapper(HuggingFaceModelWrapper):
    """
    A wrapper for the SegFormer semantic segmentation model from Hugging Face Transformers.

    This class encapsulates loading, running inference, label remapping, and visualization
    (palette colorization) for semantic segmentation tasks using SegFormer.

    Attributes
    ----------
    model_name : str
        The Hugging Face model identifier (e.g.,
        "nvidia/segformer-b5-finetuned-ade-640-640").
    device : str
        The compute device used for inference. Automatically detected if not provided.
    model : Optional[SegformerForSemanticSegmentation]
        The actual pretrained model object loaded via Hugging Face Transformers.
    processor : Optional[AutoImageProcessor]
        The associated processor for pre-processing input images.
    _ade_palette : list[int]
        The ADE20K RGB color palette used for segmentation visualization.
    """

    def __init__(self, 
                 model_name: str, 
                 device: str | None = None,
                 palette_path:str|None = None) -> None:
        """
        Initialize the SegformerSegmentationWrapper.

        Parameters
        ----------
        model_name : str
            The Hugging Face model ID or local path.
        device : Optional[str]
            The target device ("cuda", "mps", or "cpu"). If None,
            it will be auto-detected.
        """
        self.model_name = model_name
        self.device = device or self._resolve_device()
        self.model: SegformerForSemanticSegmentation | None = None
        self.processor: AutoImageProcessor | None = None
        if(palette_path is None):
            self._ade_palette: list[int] = self._load_palette_from_json("src/imageable/assets/ade20k_palette.json")
        else:
            self._ade_palette: list[int] = self._load_palette_from_json(palette_path)

    def load_model(self) -> None:
        """
        Load both the image processor and the SegFormer model onto the
        selected device.
        """
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        self.model.to(self.device)

    def is_loaded(self) -> bool:
        """
        Check whether the processor and model are successfully loaded.

        Returns
        -------
        bool
            True if both components are loaded, False otherwise.
        """
        return self.model is not None and self.processor is not None

    def preprocess(self, image: Image.Image | np.ndarray[Any, np.dtype[np.float64]]) -> dict[str, torch.Tensor]:
        """
        Preprocess image for SegFormer model.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray
            Input image to preprocess.

        Returns
        -------
        dict[str, torch.Tensor]
            Preprocessed inputs ready for the model.
        """
        if not self.is_loaded():
            self.load_model()
        assert self.processor is not None

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Store original size for later use
        self._original_size = image.size

        # Process with HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt")

        return {k: v.to(self.device) for k, v in inputs.items()}

    def postprocess(self, outputs: Any) -> np.ndarray[Any, np.dtype[np.int64]]:
        """
        Postprocess model outputs to get segmentation map.

        Parameters
        ----------
        outputs : Any
            Raw model outputs.

        Returns
        -------
        np.ndarray
            Segmentation map with class indices.
        """
        logits = outputs.logits

        # Resize logits to original image size
        resized_logits = torch.nn.functional.interpolate(
            logits,
            size=self._original_size[::-1],  # PIL uses (width, height), torch uses (height, width)
            mode="bicubic",
            align_corners=False,
        )

        # Get predictions
        prediction = torch.argmax(resized_logits, dim=1).squeeze().cpu().numpy()
        return cast("np.ndarray[Any, np.dtype[np.int64]]", prediction)

    def predict(
        self, image: Image.Image | np.ndarray[Any, np.dtype[np.float64]]
    ) -> np.ndarray[Any, np.dtype[np.int64]]:
        """
        Perform semantic segmentation on a single image.

        Parameters
        ----------
        image : PIL.Image.Image or np.ndarray
            Input image in RGB format.

        Returns
        -------
        np.ndarray
            A 2D array of class indices corresponding to the segmentation map.
        """
        if not self.is_loaded():
            self.load_model()
        assert self.model is not None

        # Preprocess
        inputs = self.preprocess(image)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        return self.postprocess(outputs)

    def remap_labels(
        self,
        seg: np.ndarray[Any, np.dtype[np.int_]],
        mapping: dict[int, int],
    ) -> np.ndarray[Any, np.dtype[np.int_]]:
        """
        Remap segmentation class IDs based on a custom mapping.

        Parameters
        ----------
        seg : np.ndarray
            A 2D segmentation map of class indices.
        mapping : dict
            Dictionary mapping source class indices to target indices.

        Returns
        -------
        np.ndarray
            A remapped version of the segmentation map.
        """
        return self._remap_labels(seg, mapping)

    def colorize(self, seg: np.ndarray[Any, np.dtype[np.int_]]) -> Image.Image:
        """
        Convert a segmentation map into a colorized image using the ADE palette.

        Parameters
        ----------
        seg : np.ndarray
            A 2D array of segmentation class indices.

        Returns
        -------
        PIL.Image.Image
            A color image in mode "P" using the predefined ADE palette.
        """
        return self._apply_palette(seg, self._ade_palette)

    def _resolve_device(self) -> str:
        """
        Automatically detect the best available compute device.

        Returns
        -------
        str
            "cuda", "mps", or "cpu"
        """
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _remap_labels(
        self,
        seg: np.ndarray[Any, np.dtype[np.int_]],
        mapping: dict[int, int],
    ) -> np.ndarray[Any, np.dtype[np.int_]]:
        """
        Remap class labels.

        Parameters
        ----------
        seg : np.ndarray
            Segmentation result to be remapped.
        mapping : dict
            Dictionary of class index mappings.

        Returns
        -------
        np.ndarray
            Remapped segmentation map.
        """
        remapped = np.zeros_like(seg)
        for src, tgt in mapping.items():
            remapped[seg == src] = tgt
        return remapped

    def _apply_palette(
        self,
        seg: np.ndarray[Any, np.dtype[np.int_]],
        palette: list[int],
    ) -> Image.Image:
        """
        Apply a color palette to a segmentation map.

        Parameters
        ----------
        seg : np.ndarray
            2D label map.
        palette : list[int]
            Flat RGB color palette list (e.g., ADE20K style).

        Returns
        -------
        PIL.Image.Image
            Colorized image using mode "P".
        """
        img = Image.fromarray(seg.astype("uint8"), mode="P")
        img.putpalette(palette)
        return img

    def _load_palette_from_json(self, path: str) -> list[int]:
        """
        Load the ADE20K color palette from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON file containing the flat list of RGB values.

        Returns
        -------
        list[int]
            The loaded RGB color palette.
        """
        path_obj = Path(path)
        with path_obj.open(encoding="utf-8") as f:
            data = json.load(f)
        return list(map(int, data))
