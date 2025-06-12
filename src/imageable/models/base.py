# src/imageable/models/base.py
from abc import ABC, abstractmethod
from typing import Any


class BaseModelWrapper(ABC):
    """
    Abstract base class for all model wrappers.
    Provides a consistent interface for loading and using models.

    Note: preprocess and postprocess are optional methods that can be
    overridden by subclasses if needed.
    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the model wrapper."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the model from checkpoint or initialize it."""

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """
        Make predictions using the loaded model.

        Parameters
        ----------
        inputs : Any
            Input data suitable for the model type.

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

    def preprocess(self, inputs: Any) -> Any:
        """
        Preprocess inputs before passing to the model.

        Default implementation returns inputs unchanged.
        Override this method in subclasses if preprocessing is needed.

        Parameters
        ----------
        inputs : Any
            Raw input data.

        Returns
        -------
        Any
            Preprocessed data ready for model inference.
        """
        return inputs

    def postprocess(self, outputs: Any) -> Any:
        """
        Postprocess model outputs.

        Default implementation returns outputs unchanged.
        Override this method in subclasses if postprocessing is needed.

        Parameters
        ----------
        outputs : Any
            Raw model outputs.

        Returns
        -------
        Any
            Processed outputs in a more usable format.
        """
        return outputs
