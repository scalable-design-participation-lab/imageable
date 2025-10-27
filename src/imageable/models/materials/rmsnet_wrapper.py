# imageable/models/materials/rmsnet_wrapper.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from imageable.models.base import BaseModelWrapper
from imageable.models.materials.rmsnet import RMSNet


class RMSNetSegmentationWrapper(BaseModelWrapper):
    def __init__(
        self,
        backbone: str = "mit_b2",
        num_classes: int = 20,
        device: str | None = None,
        sync_bn: bool = False,
        weights_path: Path | None = None,
        tile_size: int = 640,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        model_path: str = None,
    ) -> None:
        self.backbone = backbone
        self.num_classes = num_classes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sync_bn = sync_bn
        self.weights_path = Path(weights_path) if weights_path else None
        self.tile_size = tile_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.model: torch.nn.Module | None = None
        self._transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.normalize_mean, std=self.normalize_std),
            ]
        )
        self._last_size: tuple[int, int] | None = None
        self.model_path = model_path

    def load_model(self) -> None:
        if self.model is not None:
            return
        model = RMSNet(
            num_classes=self.num_classes, backbone=self.backbone, sync_bn=self.sync_bn, model_path=self.model_path
        )
        if self.weights_path is not None:
            state = torch.load(str(self.weights_path), map_location=self.device)
            model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model

    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess(self, inputs: Any) -> torch.Tensor:
        if not self.is_loaded():
            self.load_model()

        if isinstance(inputs, np.ndarray):
            img = Image.fromarray(inputs.astype(np.uint8))
        elif isinstance(inputs, Image.Image):
            img = inputs
        else:
            raise TypeError("inputs must be np.ndarray or PIL.Image")

        if img.size != (self.tile_size, self.tile_size):
            img = img.resize((self.tile_size, self.tile_size), Image.BILINEAR)

        self._last_size = (img.height, img.width)
        tensor = self._transform(img).unsqueeze(0).to(self.device)  # [1,C,H,W]
        return tensor

    def predict(self, inputs: Any) -> Any:
        if not self.is_loaded():
            self.load_model()
        x = self.preprocess(inputs)
        with torch.no_grad():
            logits = self.model(x)  # [1,C,H,W]
        return logits

    def postprocess(self, outputs: Any) -> dict[str, Any]:
        logits: torch.Tensor = outputs
        if self._last_size and logits.shape[-2:] != self._last_size:
            logits = F.interpolate(logits, size=self._last_size, mode="bilinear", align_corners=False)
        mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return {"mask": mask, "logits": logits.squeeze(0).cpu().numpy()}
