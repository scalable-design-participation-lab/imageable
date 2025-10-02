from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# -----------------------------
# ADE20K label list (indexable)
# -----------------------------
# Matches the label order used in common DPT ADE20K checkpoints (150 classes).
ADE20K_LABELS: Tuple[str, ...] = (
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'window',
    'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain',
    'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house',
    'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
    'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'dresser',
    'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
    'stairs', 'runway', 'case', 'pooltable', 'pillow', 'screen', 'stairway', 'river', 'bridge',
    'bookcase', 'blind', 'coffeetable', 'toilet', 'flower', 'book', 'hill', 'bench',
    'countertop', 'stove', 'palmtree', 'kitchen', 'computer', 'swivelchair', 'boat', 'bar',
    'arcade', 'hut', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
    'streetlight', 'booth', 'television', 'airplane', 'dirttrack', 'apparel', 'pole', 'land',
    'balustrade', 'escalator', 'ottoman', 'bottle', 'sideboard', 'poster', 'stage', 'van', 'ship',
    'fountain', 'conveyerbelt', 'canopy', 'washer', 'toy', 'pool', 'stool', 'barrel', 'basket',
    'waterfall', 'tent', 'bag', 'motorbike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
    'brandname', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
    'sculpture', 'hood', 'sconce', 'vase', 'trafficlight', 'tray', 'trashcan', 'fan', 'pier',
    'crtscreen', 'plate', 'monitor', 'bulletinboard', 'shower', 'radiator', 'glass', 'clock', 'flag'
)


# -----------------------------
# Config / thresholds
# -----------------------------

@dataclass
class DPTGateConfig:
    """
    Configuration for ADE20K gating using a DPT segmentation backend.

    Attributes
    ----------
    model_type : str
        A hint string for the backend (e.g., "dpt_hybrid"). Not used internally unless you
        implement an auto-loader.
    checkpoint_path : Optional[Path]
        Path to the DPT ADE20K checkpoint, if you implement auto-loading.
    device : str
        Torch device string ("cuda", "cpu", "mps").
    net_size : int
        Square inference size for the backend; inputs will be resized to (net_size, net_size)
        if your predictor expects that. The output will be resized back to the original size.
    thresholds : Dict[str, float]
        Per-class score thresholds for producing boolean masks.
        Keys used here: "building", "road", "sidewalk". The "building" mask is formed by OR-ing
        multiple ADE20K classes: {"house", "skyscraper", "building"} using this same threshold.
    """
    model_type: str = "dpt_hybrid"
    checkpoint_path: Optional[Path] = None
    device: str = "cpu"
    net_size: int = 480
    thresholds: Dict[str, float] = None  # set in __post_init__

    def __post_init__(self) -> None:
        if self.thresholds is None:
            # Defaults taken from your original notebook snippet
            self.thresholds = {
                "building": 15.0,
                "road": 11.0,
                "sidewalk": 12.0,
            }


# -----------------------------
# Helper utilities
# -----------------------------

def _to_pil_rgb(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected np.ndarray of shape (H,W,3); got {img.shape}")
        if img.dtype != np.uint8:
            # If float, assume [0,1]; clamp/convert to uint8
            if np.issubdtype(img.dtype, np.floating):
                arr = np.clip(img, 0.0, 1.0)
                img = (arr * 255.0 + 0.5).astype(np.uint8)
            else:
                raise ValueError(f"Unsupported dtype {img.dtype}; use uint8 or float in [0,1]")
        return Image.fromarray(img, mode="RGB")
    raise TypeError(f"Unsupported image type {type(img)}")


def _find_label_idx(name: str) -> int:
    try:
        return ADE20K_LABELS.index(name)
    except ValueError:
        raise KeyError(f"Label '{name}' not found in ADE20K label list")


def _resize_logits_to(
    logits: np.ndarray,  # (C,h,w) float32
    size_hw: Tuple[int, int]
) -> np.ndarray:
    """
    Resize per-class logits from (C,h,w) to (C,H,W) using bilinear interpolation via torch.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (C,h,w); got {logits.shape}")
    C, h, w = logits.shape
    H, W = size_hw
    t = torch.from_numpy(logits[None, ...])  # (1,C,h,w)
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t.squeeze(0).cpu().numpy()  # (C,H,W)


# -----------------------------
# DPTGate main class
# -----------------------------

class DPTGate:
    """
    A lightweight, pluggable "gate" around an ADE20K DPT semantic segmentation model.

    This class DOES NOT include an embedded DPT model implementation by default.
    You can either:
      1) Pass a `predictor` callable that takes a PIL RGB Image and returns
         per-class logits as a numpy array shaped (C,H,W), with C=150 (ADE20K).
      2) Extend `load()` to construct your DPT model and set `self.predictor`.

    If no predictor is provided, `predict` will return all-False masks (no gating),
    which is safe and keeps your pipeline functional.
    """

    def __init__(
        self,
        config: DPTGateConfig,
        predictor: Optional[Callable[[Image.Image], np.ndarray]] = None,
    ) -> None:
        """
        Parameters
        ----------
        config : DPTGateConfig
            Configuration (device, thresholds, net_size, etc.).
        predictor : Optional[Callable[[PIL.Image.Image], np.ndarray]]
            A callable that returns ADE20K logits (C,H,W) for a given RGB image.
            If None, gate acts as a no-op (all-False masks).
        """
        self.cfg = config
        self.predictor = predictor  # user-supplied or set by self.load()

        # Cache commonly used class indices
        self.idx_building = _find_label_idx("building")
        self.idx_house = _find_label_idx("house")
        self.idx_skyscraper = _find_label_idx("skyscraper")
        self.idx_road = _find_label_idx("road")
        self.idx_sidewalk = _find_label_idx("sidewalk")

    # --------------- Optional auto-loader --------------- #
    def load(self) -> None:
        """
        Optional hook to construct your DPT backend and set `self.predictor`.
        Left unimplemented on purpose to keep this module dependency-free.

        Example (pseudo-code):
        ----------------------
        model = DPTSegmentationModel(150, path=self.cfg.checkpoint_path, backbone="vitb_rn50_384").to(self.cfg.device).eval()
        tx = Compose([Resize(...), NormalizeImage(...), PrepareForNet()])
        def _pred(pil: Image.Image) -> np.ndarray:
            inp = tx({"image": np.array(pil)})["image"]
            with torch.no_grad():
                out = model(torch.from_numpy(inp).unsqueeze(0).to(self.cfg.device))
                out = F.interpolate(out, size=pil.size[::-1], mode="bicubic", align_corners=False)
            return out.squeeze(0).cpu().numpy()  # (C,H,W)
        self.predictor = _pred
        """
        # Intentionally a no-op.
        return

    # --------------- Public API --------------- #
    def predict(
        self,
        image: Union[Image.Image, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Produce boolean gating masks for keys: {'building','road','sidewalk'}.

        If no predictor is available, returns all False masks (no gating).

        Parameters
        ----------
        image : PIL.Image.Image | np.ndarray
            RGB image, any size. If ndarray, expected shape (H,W,3), dtype uint8 or float in [0,1].

        Returns
        -------
        Dict[str, np.ndarray]
            Boolean masks of shape (H,W) for 'building', 'road', and 'sidewalk'.
        """
        pil = _to_pil_rgb(image)
        H, W = pil.height, pil.width

        # No predictor: return safe "no gating" masks
        if self.predictor is None:
            false_mask = np.zeros((H, W), dtype=bool)
            return {
                "building": false_mask,
                "road": false_mask,
                "sidewalk": false_mask,
            }

        # Run predictor to get (C,h,w) logits (numpy float)
        logits = self.predictor(pil)  # expect (C,h,w)
        if not isinstance(logits, np.ndarray) or logits.ndim != 3:
            raise RuntimeError("predictor must return a numpy array of shape (C,h,w)")

        C, h, w = logits.shape
        if C < max(self.idx_building, self.idx_house, self.idx_skyscraper, self.idx_road, self.idx_sidewalk) + 1:
            raise RuntimeError(
                f"predictor returned {C} channels, but ADE20K indices up to {self.idx_skyscraper} are required"
            )

        # Resize logits back to (H,W)
        if (h, w) != (H, W):
            logits = _resize_logits_to(logits, (H, W))  # (C,H,W)

        # Thresholds
        th_building = float(self.cfg.thresholds.get("building", 15.0))
        th_road = float(self.cfg.thresholds.get("road", 11.0))
        th_sidewalk = float(self.cfg.thresholds.get("sidewalk", 12.0))

        # Building mask is an OR of several relevant ADE20K classes
        m_building = (logits[self.idx_building] > th_building) | \
                     (logits[self.idx_house] > th_building) | \
                     (logits[self.idx_skyscraper] > th_building)

        # Road
        m_road = logits[self.idx_road] > th_road

        # Sidewalk
        m_sidewalk = logits[self.idx_sidewalk] > th_sidewalk

        # Ensure boolean dtype
        return {
            "building": np.asarray(m_building, dtype=bool),
            "road": np.asarray(m_road, dtype=bool),
            "sidewalk": np.asarray(m_sidewalk, dtype=bool),
        }