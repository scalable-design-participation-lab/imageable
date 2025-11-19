# imageable/models/materials/rmsnet_wrapper.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from imageable._models.base import BaseModelWrapper
from imageable._models.materials.rmsnet import RMSNet
from huggingface_hub import hf_hub_download,try_to_load_from_cache


class RMSNetSegmentationWrapper(BaseModelWrapper):
    MODEL_REPO = "urilp4669/Material_Segmentation_Models"
    RMSN_WEIGHTS_FILENAME = "rmsnet_split2 (1).pth"
    BACKBONE_FILENAME = "mit_b2.pth"
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
        verbose: bool = False
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
        self.verbose = verbose

        if(model_path is None or weights_path is None):
            #Download weights from huggingface
            weights_path, model_path = self._download_weights()
            self.model_path = Path(model_path)
            self.weights_path = Path(weights_path)


    def _download_weights(self) -> str:
        # Try to load from cache first
        rms_cached = try_to_load_from_cache(
            repo_id=self.MODEL_REPO,
            filename=self.RMSN_WEIGHTS_FILENAME,
        )
        backbone_cached = try_to_load_from_cache(
            repo_id=self.MODEL_REPO,
            filename=self.BACKBONE_FILENAME,
        )

        if self.verbose:
            print(f"RMSNet cache: {rms_cached}")
            print(f"Backbone cache: {backbone_cached}")

        # If either is missing, download it
        rms_weights_path = rms_cached or hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename=self.RMSN_WEIGHTS_FILENAME,
        )
        model_path = backbone_cached or hf_hub_download(
            repo_id=self.MODEL_REPO,
            filename=self.BACKBONE_FILENAME,
        )

        if self.verbose:
            print(f"RMSNet weights available at: {rms_weights_path}")
            print(f"Backbone model available at: {model_path}")

        return rms_weights_path, model_path

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
