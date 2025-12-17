# imageable/models/materials/postprocess.py
from __future__ import annotations

import numpy as np


def colorize_mask(
    mask: np.ndarray,
    palette: np.ndarray,
    *,
    background_id: int | None = None,
    background_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Convert a (H, W) integer mask into an (H, W, 3) uint8 RGB image using an (N, 3) palette.

    Parameters
    ----------
    mask : np.ndarray
        2D array (H, W) with class ids in [0, N-1].
    palette : np.ndarray
        Array of shape (N, 3) with uint8 RGB rows.
    background_id : Optional[int], default=None
        If provided, any pixel with this class id will be colored with `background_color`
        (or black if `background_color` is None), without indexing into the palette.
    background_color : Optional[Tuple[int, int, int]], default=None
        RGB color for `background_id`. If None and `background_id` is provided, uses (0,0,0).

    Returns
    -------
    np.ndarray
        Color image of shape (H, W, 3), dtype uint8.

    Raises
    ------
    ValueError
        If shapes/dtypes are invalid or mask contains ids outside palette range.
    """
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a numpy array")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W); got shape {mask.shape}")

    if not isinstance(palette, np.ndarray):
        raise ValueError("palette must be a numpy array")
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError(f"palette must have shape (N,3); got {palette.shape}")
    if palette.dtype != np.uint8:
        palette = palette.astype(np.uint8, copy=False)

    h, w = mask.shape
    flat = mask.reshape(-1)

    if background_id is not None:
        # Validate all other ids are within range
        valid = (flat == background_id) | ((flat >= 0) & (flat < palette.shape[0]))
        if not np.all(valid):
            bad = np.unique(flat[~valid])
            raise ValueError(f"mask contains class ids outside palette range: {bad.tolist()}")

        colored = palette[np.clip(flat, 0, palette.shape[0] - 1)].astype(np.uint8)
        if background_color is None:
            background_color = (0, 0, 0)
        colored[flat == background_id] = np.array(background_color, dtype=np.uint8)
        return colored.reshape(h, w, 3)

    # No background override; enforce strict indexing
    if flat.min() < 0 or flat.max() >= palette.shape[0]:
        raise ValueError(
            f"mask ids must be within [0, {palette.shape[0] - 1}]; found min={int(flat.min())}, max={int(flat.max())}"
        )
    return palette[flat].reshape(h, w, 3)


def apply_gate_masks(
    materials_mask: np.ndarray,
    gate: dict[str, np.ndarray] | None,
    *,
    logic: str = "mask_keep",
    outside_id: int = 0,
) -> np.ndarray:
    """
    Apply external boolean gates (e.g., ADE20K building/road/sidewalk) to a materials mask.

    Parameters
    ----------
    materials_mask : np.ndarray
        2D array (H, W) of class ids (uint8/int).
    gate : Optional[Dict[str, np.ndarray]]
        Dict of boolean masks (H, W) with any subset of keys {"building","road","sidewalk"}.
        If None or empty, the input mask is returned unchanged.
    logic : str, default="mask_keep"
        How to combine gates:
          - "mask_keep": keep labels where ANY gate is True; set outside to `outside_id`.
          - "mask_zero_out": alias of "mask_keep" (kept for compatibility).
    outside_id : int, default=0
        Class id to assign outside allowed regions.

    Returns
    -------
    np.ndarray
        Gated mask (H, W), same dtype as input.

    Raises
    ------
    ValueError
        If shapes are inconsistent or logic is unknown.
    """
    if gate is None or len(gate) == 0:
        return materials_mask

    if materials_mask.ndim != 2:
        raise ValueError(f"materials_mask must be 2D (H,W); got {materials_mask.shape}")

    h, w = materials_mask.shape

    # Combine provided gates into a single boolean "allow" mask using OR.
    allow = None
    for key in ("building", "road", "sidewalk"):
        m = gate.get(key)
        if m is None:
            continue
        if not isinstance(m, np.ndarray) or m.dtype != bool or m.shape != (h, w):
            raise ValueError(
                f"gate['{key}'] must be a boolean array with shape (H,W)={materials_mask.shape}; got {None if m is None else (m.dtype, m.shape)}"
            )
        allow = m if allow is None else (allow | m)

    if allow is None:
        # No usable gates present -> return unchanged
        return materials_mask

    logic = logic.lower().strip()
    if logic not in {"mask_keep", "mask_zero_out"}:
        raise ValueError(f"Unknown gating logic '{logic}'. Use 'mask_keep' or 'mask_zero_out'.")

    out = materials_mask.copy()
    out[~allow] = outside_id
    return out
