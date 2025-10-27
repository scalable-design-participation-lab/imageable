from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class MaterialsDetectionResult:
    """
    Container for material segmentation results.

    Attributes
    ----------
    mask : np.ndarray
        2D array (H,W) of integer class IDs (0..N-1).
    colored : Optional[np.ndarray]
        3D array (H,W,3) of RGB visualization. None if not requested.
    logits : Optional[torch.Tensor]
        Raw model logits before argmax. Shape (C,H,W). None if not requested.
    meta : dict
        Extra metadata (image size, thresholds, device, etc.).
    """

    mask: np.ndarray
    colored: np.ndarray | None = None
    logits: torch.Tensor | None = None
    meta: dict | None = None

    def to_pil(self) -> Image.Image | None:
        """
        Convert the colored mask (if available) to a PIL Image.

        Returns
        -------
        Optional[Image.Image]
            A PIL.Image.Image if `colored` is set, else None.
        """
        if self.colored is None:
            return None
        return Image.fromarray(self.colored.astype("uint8"))

    def save(self, path: str) -> None:
        """
        Save the colored mask to disk.

        Parameters
        ----------
        path : str
            Path to save the image.
        """
        if self.colored is None:
            raise RuntimeError("No colored mask available to save.")
        img = self.to_pil()
        if img is not None:
            img.save(path)
