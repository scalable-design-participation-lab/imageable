


from typing import Any

import joblib
import numpy as np
from huggingface_hub import hf_hub_download, try_to_load_from_cache

from imageable._models.base import BaseModelWrapper
from imageable._models.height_correction_model import HeightCorrectionModel


class LineParameterSelectionModel(BaseModelWrapper):
    MODEL_REPO = "walup/cluster_height_correction_model"
    VALUES_FILE = "best_line_score_params_per_cluster.pkl"

    def __init__(
            self,
            height_correction_model:HeightCorrectionModel|None = None,
            cluster_param_values:list[float]|np.ndarray|None = None
            )->None:
        """
        Initialize the LineParameterSelection model.

        Parameters
        ----------
        height_correction_model
            Cluster correction model used to determine cluster based on input features.
            If no model is provided, pretrained values will be downloaded.
        cluster_param_values
            List or array of line detection parameter values corresponding to each cluster.
        """
        self.corr_model = height_correction_model
        self.param_values = cluster_param_values

    def predict(self,
                vector:list[float]|np.ndarray)->float:

        # Predict cluster
        if(self.is_loaded()):
            assert self.corr_model is not None
            assert self.corr_model.pretrained is not None
            assert self.param_values is not None
            #First scale the vector
            scaled_vector = self.corr_model.scaler.transform(vector)
            weights = self.corr_model.pretrained._compute_weights(scaled_vector)
            predicted_cluster = np.argmax(weights)

            return self.param_values[predicted_cluster]
        raise RuntimeError("Model not loaded. Call load_model() before predict().")
    def is_loaded(self)->bool:
        return (self.corr_model is not None and self.corr_model.is_loaded()) and (self.param_values is not None)

    def preprocess(
            self,
            inputs:Any)->Any:
        return super().preprocess(inputs)

    def postprocess(
            self,
            outputs:Any)->Any:
        return super().postprocess(outputs)

    def load_model(self)->None:
        if(not self.is_loaded()):
            self.corr_model = HeightCorrectionModel()
            self.corr_model.load_model()
            self._download_param_values()

    def _download_param_values(self)->None:

        cached = try_to_load_from_cache(
            repo_id=self.MODEL_REPO,
            filename=self.VALUES_FILE,
        )

        if cached is None:
            file_path = hf_hub_download(
                repo_id=self.MODEL_REPO,
                filename=self.VALUES_FILE,
            )
        else:
            file_path = cached

        if file_path is None:
            raise ValueError("Could not download model from huggingface")
        self.param_values = joblib.load(file_path)
