# imageable/models/materials/preprocess.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    """
    Configuration for materials image preprocessing.

    Attributes
    ----------
    tile_size : int
        Target square size. Inputs will be resized to (tile_size, tile_size).
    mean : Tuple[float, float, float]
        Per-channel normalization mean in RGB order.
    std : Tuple[float, float, float]
        Per-channel normalization std in RGB order.
    to_float32 : bool
        If True, convert PIL/np inputs to float32 before tensor conversion.
    """

    tile_size: int = 640
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    to_float32: bool = False


def _to_pil(img: Image.Image | np.ndarray) -> Image.Image:
    """
    Ensure input is a PIL.Image in RGB mode.
    Accepts: PIL.Image (any mode) or numpy array (H,W,3) uint8/float.
    """
    if isinstance(img, Image.Image):
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    if isinstance(img, np.ndarray):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"numpy image must be (H,W,3), got {img.shape}")

        if img.dtype == np.uint8:
            pil = Image.fromarray(img, mode="RGB")
            return pil
        if np.issubdtype(img.dtype, np.floating):
            # clamp to [0,1] then scale to [0,255]
            arr = np.clip(img, 0.0, 1.0)
            arr = (arr * 255.0 + 0.5).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        raise ValueError(f"Unsupported numpy dtype {img.dtype} (expected uint8 or float)")

    raise TypeError(f"Unsupported image type {type(img)}; expected PIL.Image or numpy.ndarray")


def _ensure_tile(img: Image.Image, tile_size: int) -> Image.Image:
    """
    Force image to (tile_size, tile_size) with bilinear resize.
    """
    if img.size != (tile_size, tile_size):
        return img.resize((tile_size, tile_size), Image.BILINEAR)
    return img


def _build_transform(mean: tuple[float, float, float], std: tuple[float, float, float]) -> T.Compose:
    """
    Torchvision transform: ToTensor -> Normalize (RGB).
    """
    return T.Compose(
        [
            T.ToTensor(),  # (H,W,3) uint8 -> (C,H,W) float32 in [0,1]
            T.Normalize(mean=mean, std=std),
        ]
    )


@overload
def preprocess_image(
    img: Image.Image | np.ndarray,
    config: PreprocessConfig = ...,
    device: str | torch.device | None = ...,
) -> torch.Tensor: ...
@overload
def preprocess_image(
    img: Sequence[Image.Image | np.ndarray],
    config: PreprocessConfig = ...,
    device: str | torch.device | None = ...,
) -> torch.Tensor: ...


def preprocess_image(
    img: Image.Image | np.ndarray | Sequence[Image.Image | np.ndarray],
    config: PreprocessConfig = PreprocessConfig(),
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Preprocess a single image OR a sequence of images for RMSNet.

    - Converts to RGB PIL
    - Resizes to (tile_size, tile_size)
    - ToTensor + Normalize with ImageNet stats
    - Adds batch dimension

    Parameters
    ----------
    img : PIL.Image | np.ndarray | Sequence[...]
        Single image or a list/tuple of images.
    config : PreprocessConfig
        Preprocessing configuration.
    device : str | torch.device | None
        If provided, move the output tensor to this device.

    Returns
    -------
    torch.Tensor
        For single image: shape (1, 3, tile_size, tile_size)
        For sequence:     shape (N, 3, tile_size, tile_size)
    """
    transform = _build_transform(config.mean, config.std)

    def _one(x: Image.Image | np.ndarray) -> torch.Tensor:
        pil = _to_pil(x)
        pil = _ensure_tile(pil, config.tile_size)
        tensor = transform(pil)  # (C,H,W)
        return tensor

    if isinstance(img, (list, tuple)):
        tensors: list[torch.Tensor] = [_one(x) for x in img]
        batch = torch.stack(tensors, dim=0)  # (N,C,H,W)
    else:
        batch = _one(img).unsqueeze(0)  # (1,C,H,W)

    if device is not None:
        return batch.to(device)
    return batch


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Inverse of Normalize for visualization/debug.

    Parameters
    ----------
    tensor : torch.Tensor
        (N,C,H,W) normalized tensor.
    mean, std : tuple
        Per-channel stats used during normalization.

    Returns
    -------
    torch.Tensor
        Denormalized tensor in [0,1] (clamped).
    """
    if tensor.ndim != 4 or tensor.shape[1] != 3:
        raise ValueError(f"Expected (N,3,H,W), got {tuple(tensor.shape)}")

    m = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)[None, :, None, None]
    s = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)[None, :, None, None]
    out = tensor * s + m
    return out.clamp_(0.0, 1.0)


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a single image tensor (3,H,W) or batch (1,3,H,W) in [0,1]
    to a numpy array (H,W,3) uint8 for quick visualization.

    Parameters
    ----------
    tensor : torch.Tensor
        (3,H,W) or (1,3,H,W) tensor in [0,1].

    Returns
    -------
    np.ndarray
        (H,W,3) uint8 RGB.
    """
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected (3,H,W) or (1,3,H,W), got {tuple(tensor.shape)}")

    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return arr
