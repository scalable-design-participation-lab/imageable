# imageable/models/materials/rmsnet.py
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from imageable._models.materials.backbones.segformer_mit_encoder import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from imageable._models.materials.decoders.samixer_head import SAMixerHead

try:
    # optional; only used if you enable sync_bn
    from imageable._models.materials.norms.sync_batchnorm import SynchronizedBatchNorm2d
except Exception:
    SynchronizedBatchNorm2d = None  # type: ignore

_BACKBONES = {
    "mit_b0": mit_b0,
    "mit_b1": mit_b1,
    "mit_b2": mit_b2,
    "mit_b3": mit_b3,
    "mit_b4": mit_b4,
    "mit_b5": mit_b5,
}


class RMSNet(nn.Module):
    """
    Segmentation model: SegFormer MIT backbone + SAMixer head.
    Returns logits [B, num_classes, H, W].
    """

    def __init__(
        self,
        num_classes: int = 20,
        backbone: str = "mit_b2",
        sync_bn: bool = False,
        freeze_bn: bool = False,
        model_path=None,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(_BACKBONES)}")

        self.backbone = _BACKBONES[backbone](model_path=model_path)  # returns list of 4 feature maps
        norm_layer = SynchronizedBatchNorm2d if (sync_bn and SynchronizedBatchNorm2d is not None) else nn.BatchNorm2d

        self.decoder = SAMixerHead(
            in_channels=[64, 128, 320, 512],
            feature_strides=[4, 8, 16, 32],
            embedding_dim=768,
            norm_layer=norm_layer,
            num_classes=num_classes,
            in_index=[0, 1, 2, 3],
            dropout_ratio=0.1,
            input_transform="multiple_select",
            align_corners=False,
        )

        self._freeze_bn = freeze_bn
        self.model_path = model_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # features: [C1, C2, C3, C4] at strides [4,8,16,32]
        feats = self.backbone(x)
        logits = self.decoder(feats)  # typically stride-4 resolution
        # upsample to input size
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits

    def freeze_bn(self) -> None:
        if not self._freeze_bn:
            return
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d,)) or (
                SynchronizedBatchNorm2d is not None and isinstance(m, SynchronizedBatchNorm2d)
            ):
                m.eval()
