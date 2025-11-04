import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imageable.models.base import BaseModelWrapper
from pathlib import Path
from huggingface_hub import hf_hub_download, try_to_load_from_cache

class DistanceRegressorWrapper(BaseModelWrapper):
    MODEL_REPO = "urilp4669/footprint_distance_to_nearest_street"
    CKPT_FILENAME = "distance_regressor.pth"

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        device: str | None = None,
        model_path: str | None = None,
        target_mode: str = "standard"
    ) -> None:
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.device = device or self._resolve_device()
        self.ckpt_path = model_path
        self.model: nn.Module | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._loaded = False

        self._target_mode: str = target_mode
        self._y_mean: torch.Tensor | None = None
        self._y_std: torch.Tensor | None = None

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build(self) -> nn.Module:
        layers = []
        in_dim = self.input_dim
        for h in self.hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        return nn.Sequential(*layers)

    def _resolve_ckpt_path(self, force_download: bool = False) -> str | None:
        try:
            if not force_download:
                cached = try_to_load_from_cache(
                    repo_id=self.MODEL_REPO,
                    filename=self.CKPT_FILENAME,
                )
                if cached:
                    return cached
            # either no cache or we want a fresh copy
            return hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.CKPT_FILENAME,
                force_download=force_download,  # << key
            )
        except Exception:
            return None

    def _maybe_set_valid_ckpt(self) -> None:
        """
        Resolve/download checkpoint. If input_dim mismatches, force a fresh download.
        If it STILL mismatches, give up (leave self.ckpt_path=None).
        """
        def _check_match(path: str | None) -> bool:
            if not path:
                return False
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                state = ckpt["model_state"]
                # first Linear weight in Sequential
                first_w_key = next(k for k in state.keys() if k.endswith(".weight"))
                in_dim_ckpt = state[first_w_key].shape[1]
                return in_dim_ckpt == self.input_dim
            except Exception:
                return False

        # 1) try cache / existing
        path = self.ckpt_path or self._resolve_ckpt_path(force_download=False)
        if _check_match(path):
            self.ckpt_path = path
            return

        # 2) force fresh download (bypass cache)
        fresh = self._resolve_ckpt_path(force_download=True)
        if _check_match(fresh):
            self.ckpt_path = fresh
            return

        # 3) still no match -> don’t use any ckpt (train/use scratch)
        self.ckpt_path = None


    def load_model(self) -> None:
        if self.model is None:
            self.model = self._build().to(self.device)

        if not self.ckpt_path:
            self._maybe_set_valid_ckpt()

        if self.ckpt_path:
            ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state"])
            self._mean = torch.as_tensor(ckpt["x_mean"], dtype=torch.float32, device=self.device)
            self._std  = torch.as_tensor(ckpt["x_std"],  dtype=torch.float32, device=self.device)

            # load target transform state (if present)
            self._target_mode = ckpt.get("target_mode", "none")
            ymean = ckpt.get("y_mean", None)
            ystd  = ckpt.get("y_std",  None)
            self._y_mean = torch.as_tensor(ymean, dtype=torch.float32, device=self.device) if ymean is not None else None
            self._y_std  = torch.as_tensor(ystd,  dtype=torch.float32, device=self.device) if ystd  is not None else None

        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded and (self.model is not None)

    def preprocess(self, inputs: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        if self._mean is not None and self._std is not None:
            x = (x - self._mean) / (self._std + 1e-8)
        return x

    def _inverse_target(self, y_pred_tensor: torch.Tensor) -> np.ndarray:
        if self._target_mode == "standard":
            assert self._y_mean is not None and self._y_std is not None
            y = y_pred_tensor * (self._y_std + 1e-8) + self._y_mean
            return y.detach().cpu().numpy()
        elif self._target_mode == "log1p":
            y = torch.expm1(y_pred_tensor)
            return y.detach().cpu().numpy()
        else:
            return y_pred_tensor.detach().cpu().numpy()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if not self.is_loaded():
            self.load_model()
        assert self.model is not None
        x = self.preprocess(inputs)
        with torch.no_grad():
            y_scaled = self.model(x).squeeze(-1)
        return self._inverse_target(y_scaled)


def train_distance_regressor(
    X,
    y,
    wrapper: DistanceRegressorWrapper,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    ckpt_out: str = "distance_regressor.pth",
    val_split: float = 0.2,
    seed: int = 42,
    target_mode: str = "standard",  # "standard" | "log1p" | "none"
    loss_type: str = "huber",       # "huber" | "mse" | "mae"
    loss_scale: float = 1.0         # >1.0 helps tiny targets if using "none"
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    # feature normalization
    x_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    x_std  = X.std(axis=0, keepdims=True).astype(np.float32)
    x_std[x_std == 0.0] = 1.0
    Xn = (X - x_mean) / x_std

    # target transform
    if target_mode == "standard":
        y_mean = y.mean().astype(np.float32)
        y_std  = y.std().astype(np.float32)
        if y_std == 0.0:
            y_std = np.float32(1.0)
        y_t = (y - y_mean) / (y_std + 1e-8)
        y_save_mean, y_save_std = y_mean, y_std
    elif target_mode == "log1p":
        y_t = np.log1p(y).astype(np.float32)
        y_save_mean, y_save_std = None, None
    else:  # "none"
        y_t = y.astype(np.float32)
        y_save_mean, y_save_std = None, None

    X_tr, X_va, y_tr, y_va = train_test_split(Xn, y_t, test_size=val_split, random_state=seed)

    ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).float())
    ds_va = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va).float())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)

    # set wrapper state
    wrapper._mean = torch.tensor(x_mean.squeeze(0), dtype=torch.float32, device=wrapper.device)
    wrapper._std  = torch.tensor(x_std.squeeze(0),  dtype=torch.float32, device=wrapper.device)
    wrapper._target_mode = target_mode
    wrapper._y_mean = torch.tensor(y_save_mean, dtype=torch.float32, device=wrapper.device) if y_save_mean is not None else None
    wrapper._y_std  = torch.tensor(y_save_std,  dtype=torch.float32, device=wrapper.device) if y_save_std  is not None else None

    # Build model; load ckpt only if valid (matching input_dim). Otherwise stays scratch.
    wrapper.load_model()
    model = wrapper.model
    assert model is not None

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_type == "huber":
        loss_fn = nn.SmoothL1Loss()
    elif loss_type == "mae":
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(wrapper.device)
            yb = yb.to(wrapper.device)
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            if loss_scale != 1.0:
                loss = loss * loss_scale
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in dl_va:
                xb = xb.to(wrapper.device)
                yb = yb.to(wrapper.device)
                pred = model(xb).squeeze(-1)
                vloss = loss_fn(pred, yb)
                if loss_scale != 1.0:
                    vloss = vloss * loss_scale
                val_losses.append(vloss.item())
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = model.state_dict()

    torch.save(
        {
            "model_state": best_state,
            "input_dim": wrapper.input_dim,
            "hidden_sizes": wrapper.hidden_sizes,
            "x_mean": wrapper._mean.detach().cpu().numpy(),
            "x_std":  wrapper._std.detach().cpu().numpy(),
            "target_mode": target_mode,
            "y_mean": None if wrapper._y_mean is None else wrapper._y_mean.detach().cpu().numpy().item(),
            "y_std":  None if wrapper._y_std  is None else wrapper._y_std.detach().cpu().numpy().item(),
        },
        ckpt_out,
    )
    return ckpt_out
