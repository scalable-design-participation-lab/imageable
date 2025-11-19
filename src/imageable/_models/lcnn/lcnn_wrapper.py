from curses import raw
from pathlib import Path
from typing import Any

import numpy as np
import skimage.io
import skimage.transform
import torch
import yaml
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from imageable._models.base import BaseModelWrapper
from imageable._models.lcnn import models as lcnn_models
from imageable._models.lcnn.core.config import C, M
from imageable._models.lcnn.models.line_vectorizer import LineVectorizer
from imageable._models.lcnn.models.multitask_learner import MultitaskHead, MultitaskLearner
from imageable._models.lcnn.core.postprocess import postprocess



class LCNNWrapper(BaseModelWrapper):
    """
    A wrapper for the L-CNN (Line-CNN) model for line detection in images.

    This class encapsulates loading, preprocessing, inference, and postprocessing
    for the L-CNN model which detects lines/wireframes in images.

    Attributes
    ----------
    config_path : str
        Path to the YAML configuration file.
    checkpoint_path : str
        Path to the model checkpoint file.
    device : str
        The compute device used for inference.
    model : Optional[LineVectorizer]
        The loaded L-CNN model.
    config : dict
        Model configuration loaded from YAML.
    """
    MODEL_REPO = "urilp4669/LCNN_Weights"
    DOWNLOAD_FILENAME = "190418-201834-f8934c6-lr4d10-312k.pth"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> None:
        """
        Initialize the L-CNN wrapper.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary. If not provided, will load from config_path.
        config_path : str, optional
            Path to the YAML configuration file.
        checkpoint_path : str, optional
            Path to the model checkpoint file.
        device : str, optional
            The target device ("cuda", "mps", or "cpu"). If None, it will be auto-detected.
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device or self._resolve_device()
        self.model: Any = None  # LineVectorizer type

        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            # Default configuration for wireframe detection
            self.config = {
                "model": {
                    "depth": 4,
                    "num_stacks": 2,
                    "num_blocks": 1,
                    "head_size": [[3], [1], [2]],
                    "image": {"mean": [109.730, 103.832, 98.681], "stddev": [22.275, 22.124, 23.229]},
                }
            }

        # Update global L-CNN configuration
        if "model" in self.config:
            M.update(self.config["model"])

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

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file.

        Returns
        -------
        dict
            Configuration dictionary.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Update global configuration objects used by L-CNN
        C.update(config)
        M.update(config.get("model", {}))

        return config

    def load_model(self) -> None:
        """
        Load the L-CNN model from checkpoint, checking cache first.
        """
        if not self.checkpoint_path:
            # Try to locate checkpoint in cache
            cached_path = try_to_load_from_cache(
                repo_id=self.MODEL_REPO,
                filename=self.DOWNLOAD_FILENAME,
            )

            if cached_path and Path(cached_path).exists():
                self.checkpoint_path = cached_path
                print(f"Checkpoint found in cache: {cached_path}")
            else:
                # Download if not cached
                model_path = hf_hub_download(
                    repo_id=self.MODEL_REPO,
                    filename=self.DOWNLOAD_FILENAME,
                )
                self.checkpoint_path = model_path
                print(f"Checkpoint downloaded at: {model_path}")

        # Final safety check
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found at {self.checkpoint_path}")


        # Update global configuration
        model_config = self.config.get("model", {})
        M.update(model_config)

        # Initialize model architecture using local lcnn_models
        model = lcnn_models.hourglass_pose.hg(
            depth=model_config.get("depth", 4),
            head=lambda c_in, c_out: MultitaskHead(c_in, c_out),
            num_stacks=model_config.get("num_stacks", 2),
            num_blocks=model_config.get("num_blocks", 1),
            num_classes=sum(sum(model_config.get("head_size", [[3], [1], [2]]), [])),
        )

        # Wrap with multitask learner and line vectorizer
        model = MultitaskLearner(model)
        model = LineVectorizer(model)

        # Load checkpoint
        device = torch.device(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()

        self.model = model

    def is_loaded(self) -> bool:
        """
        Check whether the model is successfully loaded.

        Returns
        -------
        bool
            True if the model is loaded, False otherwise.
        """
        return self.model is not None

    def preprocess(self, image: np.ndarray | str) -> torch.Tensor:
        """
        Preprocess an image for L-CNN inference.

        Parameters
        ----------
        image : np.ndarray or str
            Input image as numpy array or path to image file.

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor ready for inference.
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = skimage.io.imread(image)

        # Convert grayscale to RGB
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, 2)

        # Take only first 3 channels if image has alpha
        image = image[:, :, :3]

        # Store original shape for later
        self._original_shape = image.shape[:2]

        # Resize to 512x512
        im_resized = skimage.transform.resize(image, (512, 512)) * 255

        # Normalize using model's mean and stddev
        image_config = self.config["model"]["image"]
        normalized = (im_resized - image_config["mean"]) / image_config["stddev"]

        # Convert to torch tensor with correct shape [1, C, H, W]
        tensor = torch.from_numpy(np.rollaxis(normalized, 2)[None].copy()).float()

        return tensor

    def predict(self, image: np.ndarray | str) -> dict[str, Any]:
        """
        Perform line detection on an image.

        Parameters
        ----------
        image : np.ndarray or str
            Input image as numpy array or path to image file.

        Returns
        -------
        dict
            Dictionary containing:
            - 'lines': Detected line segments as an array of shape (N, 4)
            - 'scores': Confidence scores for each line as an array of shape (N,)
            - 'processed_lines': Post-processed lines
            - 'processed_scores': Post-processed scores
        """
        if not self.is_loaded():
            self.load_model()

        # 1) Preprocess (reads file if `image` is a path)
        image_tensor = self.preprocess(image)

        # 2) Build the input dict for the L-CNN model
        device = torch.device(self.device)
        input_dict = {
            "image": image_tensor.to(device),
            "meta": [
                {
                    "junc": torch.zeros(1, 2).to(device),
                    "jtyp": torch.zeros(1, dtype=torch.uint8).to(device),
                    "Lpos": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                    "Lneg": torch.zeros(2, 2, dtype=torch.uint8).to(device),
                }
            ],
            "target": {
                "jmap": torch.zeros([1, 1, 128, 128]).to(device),
                "joff": torch.zeros([1, 1, 2, 128, 128]).to(device),
            },
            "mode": "testing",
        }

        # 3) Inference
        with torch.no_grad():
            outputs = self.model(input_dict)["preds"]

        # 4) Scale raw 128×128 coords back to original image size
        h, w = self._original_shape

        raw = outputs["lines"][0].cpu().numpy()# shape (N, 4)
        if raw.ndim == 2 and raw.shape[1] == 4:
            # raw is (N,4): [y0, x0, y1, x1] or [x0, y0, x1, y1] depending on model
            # If your model uses [y, x] ordering (common in LCNN), scales should be [h, w, h, w]
            scales = np.array([h, w, h, w], dtype=float)
            raw_flat = raw / 128.0 * scales           # (N,4)
            lines_for_post = raw_flat.reshape(-1, 2, 2)
        else:
            # raw is (N,2,2): [[y,x],[y,x]] or [[x,y],[x,y]] depending on model
            # If ordering is [y, x], scale with [h, w] along the last dim:
            raw_scaled = (raw / 128.0) * np.array([h, w], dtype=float)  # (N,2,2)
            lines_for_post = raw_scaled
            raw_flat = lines_for_post.reshape(-1, 4)

        # 6) Extract scores
        scores = outputs["score"][0].cpu().numpy()    # shape (N,)

        # 7) Post-process to remove duplicates/overlaps
        processed_lines, processed_scores = self.postprocess({
            "lines": lines_for_post,
            "scores": scores,
            "image_shape": self._original_shape,
        })

        # 8) Return the flat array for your callers/tests
        return {
            "lines": raw_flat,
            "scores": scores,
            "processed_lines": processed_lines,
            "processed_scores": processed_scores,
        }

    def postprocess(self, outputs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model outputs to remove overlapped lines.

        Parameters
        ----------
        outputs : dict
            Dictionary containing 'lines', 'scores', and 'image_shape'.

        Returns
        -------
        tuple
            Processed lines and scores.
        """
        lines = outputs["lines"]
        scores = outputs["scores"]
        image_shape = outputs["image_shape"]

        # Calculate diagonal for threshold
        diag = (image_shape[0] ** 2 + image_shape[1] ** 2) ** 0.5

        # Postprocess to remove overlapped lines
        processed_lines, processed_scores = postprocess(lines, scores, diag * 0.01, 0, False)

        return processed_lines, processed_scores

    def save_results(self, results: dict[str, Any], output_path: str) -> None:
        """
        Save detection results to a file.

        Parameters
        ----------
        results : dict
            Results from predict() method.
        output_path : str
            Path where to save the results (as .npz file).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_path,
            lines=results["lines"],
            scores=results["scores"],
            processed_lines=results["processed_lines"],
            processed_scores=results["processed_scores"],
        )

    def process_batch(self, image_paths: list[str], output_dir: str) -> list[dict[str, Any]]:
        """
        Process a batch of images.

        Parameters
        ----------
        image_paths : list[str]
            List of paths to images.
        output_dir : str
            Directory where to save results.

        Returns
        -------
        list[dict]
            List of results for each image.
        """
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for img_path in image_paths:
            try:
                # Process image
                result = self.predict(img_path)

                # Save results
                base_name = Path(img_path).stem
                output_path = output_dir / f"{base_name}.npz"
                self.save_results(result, str(output_path))

                results.append({"image_path": img_path, "success": True, "result": result})

            except Exception as e:
                results.append({"image_path": img_path, "success": False, "error": str(e)})

        return results
