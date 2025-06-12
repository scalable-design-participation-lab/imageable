from abc import abstractmethod
from typing import Any

from imageable.models.base import BaseModelWrapper


class HuggingFaceModelWrapper(BaseModelWrapper):
    """
    Abstract base class for Hugging Face model wrappers.
    Provides a consistent interface for loading and using models.
    """

    @abstractmethod
    def __init__(self, model_name: str, device: str | None = None) -> None:
        """
        Initialize the model wrapper with the specified model name and device.

        Parameters
        ----------
        model_name : str
            The name or path of the Hugging Face model.
        device : str, optional
            The device to load the model onto (e.g., 'cpu' or 'cuda').
        """

    @abstractmethod
    def load_model(self) -> None:
        """Load the model from Hugging Face or local cache."""

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """
        Make predictions using the loaded model.

        Parameters
        ----------
        inputs : Any
            Input data suitable for the model type (e.g., string, dict, tensor).

        Returns
        -------
        Any
            Model-specific prediction result.

        Raises
        ------
        RuntimeError
            If the model has not been loaded.
        """

    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.

        Returns
        -------
        bool
            True if the model is loaded and ready to predict; False otherwise.
        """

    @abstractmethod
    def preprocess(self, inputs: Any) -> Any:
        """
        Preprocess inputs before passing to the model.

        Parameters
        ----------
        inputs : Any
            Raw input data.

        Returns
        -------
        Any
            Preprocessed data ready for model inference.
        """

    @abstractmethod
    def postprocess(self, outputs: Any) -> Any:
        """
        Postprocess model outputs.

        Parameters
        ----------
        outputs : Any
            Raw model outputs.

        Returns
        -------
        Any
            Processed outputs in a more usable format.
        """
