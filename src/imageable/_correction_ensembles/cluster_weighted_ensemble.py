from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
from libpysal.weights import Queen
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import StandardScaler
from spopt.region import Skater

from imageable._models.base import BaseModelWrapper


def get_best_k_silhouette(
    data: np.ndarray,
    k_range: tuple[int, int] = (2, 10),
) -> tuple[int, dict[int, float]]:
    best_k = None
    best_score = -np.inf
    k_min = int(k_range[0])
    k_max = int(k_range[1])
    scores: dict[int, float] = {}
    data = np.asarray(data, dtype=np.float64)


    if np.isnan(data).any() or np.isinf(data).any():
        msg = "Input data contains NaN or inf."
        raise ValueError(msg)

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, n_init=1).fit(data)
        labels = kmeans.labels_


        _, counts = np.unique(labels, return_counts=True)
        if np.any(counts == 0):
            continue

        s = silhouette_score(data, labels)
        scores[k] = s

        if s > best_score:
            best_score = s
            best_k = k

    if best_k is None:
        msg = f"No valid k found in range {k_range} (all had empty clusters)."
        raise ValueError(msg)

    assert isinstance(best_k, int)


    return best_k, scores


class ClusterWeightedEnsembleWrapper(BaseModelWrapper):
    """
    Wrapper for a cluster-weighted ensemble of regressors.

    - Run KMeans to partition feature space into n_clusters.
    - Train one regressor per cluster on that cluster's points.
    - At inference, each new sample is predicted by all cluster models,
      and we take a distance-based soft weighting of those per-cluster predictions.
    """

    def __init__(
        self,
        n_clusters: int,
        model_factory: Callable[[int], Any],
        kmeans_kwargs: dict[str, Any] | None = None,
        distance_eps: float = 1e-12,
        feature_indices_used_for_clustering: list[int] | None = None,
        scale: bool = False,
        decay_constant: float = 3.0,
        early_stopping_rounds: int | None = None,
        min_validation_size: int = 50,
    ) -> None:
        self.n_clusters = n_clusters
        self.model_factory = model_factory
        self.kmeans_kwargs = {} if kmeans_kwargs is None else kmeans_kwargs
        self.distance_eps = distance_eps
        self.decay_constant = decay_constant
        self.feature_indices_used_for_clustering = feature_indices_used_for_clustering
        self.scale = scale
        self.scaler = StandardScaler() if scale else None
        self.early_stopping_rounds = early_stopping_rounds
        self.min_validation_size = min_validation_size

        self._kmeans = None
        self._cluster_centers: np.ndarray | None = None
        self._cluster_models: list[Any] = []
        self._is_loaded = False

    def load_model(
        self,
        X: np.ndarray|None = None,
        y: np.ndarray|None = None,
        X_val: np.ndarray|None = None,
        y_val: np.ndarray|None = None,
    ) -> None:
        """
        Fit KMeans + per-cluster regressors.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        if(X is None or y is None):
            msg = "X and y must be provided to load_model."
            raise ValueError(msg)
        X = np.asarray(X)
        y = np.asarray(y)

        # ---- SCALE IF NEEDED ----
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        # X is now the "training X" used throughout
        X_complete = X.copy()

        # features used for clustering (maybe a subset)
        if self.feature_indices_used_for_clustering is not None:
            X_cluster = X[:, self.feature_indices_used_for_clustering]
        else:
            X_cluster = X

        # KMeans on cluster features
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            **self.kmeans_kwargs,
        )
        kmeans.fit(X_cluster)

        # ensure fitting succeeded and assign to self._kmeans
        if not hasattr(kmeans, "labels_"):
            msg = "KMeans fit did not produce labels_."
            raise RuntimeError(msg)
        self._kmeans = kmeans

        labels = kmeans.labels_
        self._cluster_centers = kmeans.cluster_centers_

        # optional early-stopping validation set, assigned to clusters the same
        # way as the training data (only used if X_val/y_val and early_stopping_rounds are given)
        val_labels = None
        if X_val is not None and y_val is not None and self.early_stopping_rounds is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
            X_val_complete = self.scaler.transform(X_val) if self.scaler is not None else X_val
            if self.feature_indices_used_for_clustering is not None:
                X_val_cluster = X_val_complete[:, self.feature_indices_used_for_clustering]
            else:
                X_val_cluster = X_val_complete
            val_labels = kmeans.predict(X_val_cluster)

        # one expert per cluster, trained on full X
        self._cluster_models = []
        for k in range(self.n_clusters):
            mask = labels == k
            model = self.model_factory(k)
            v = val_labels == k if val_labels is not None else None
            if v is not None and int(v.sum()) >= self.min_validation_size:
                model.set_params(early_stopping_rounds=self.early_stopping_rounds)
                model.fit(
                    X_complete[mask], y[mask],
                    eval_set=[(X_val_complete[v], y_val[v])],
                    verbose=False,
                )
            else:
                # too few validation points → skip early stopping, train full n_estimators
                model.fit(X_complete[mask], y[mask])
            self._cluster_models.append(model)

        self._is_loaded = True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray|None = None,
        y_val: np.ndarray|None = None,
    ) -> "ClusterWeightedEnsembleWrapper":
        self.load_model(X, y, X_val, y_val)
        return self

    def is_loaded(self) -> bool:
        return self._is_loaded

    def _compute_weights(
            self,
            X: np.ndarray) -> np.ndarray:
        """
        Compute soft assignment weights for each sample to each cluster,
        based on distance to cluster centers.

        X is assumed to be in the same feature space as training X
        (i.e., already scaled if self.scaler is not None).
        """
        # _cluster_centers is declared in __init__; just check for None
        if self._cluster_centers is None:
            msg = "Model not loaded: no cluster centers."
            raise RuntimeError(msg)

        X = np.asarray(X)

        if self.feature_indices_used_for_clustering is not None:
            X_cluster = X[:, self.feature_indices_used_for_clustering]
        else:
            X_cluster = X

        dists = pairwise_distances(X_cluster, self._cluster_centers)
        dists = np.maximum(dists, self.distance_eps)
        min_dists = np.min(dists, axis=1, keepdims=True)
        w = np.exp(-self.decay_constant * (dists - min_dists))
        w = w / np.sum(w, axis=1, keepdims=True)
        w = np.asarray(w, dtype = np.float64)
        return w

    def predict(self, inputs: Any) -> np.ndarray:
        if not self.is_loaded():
            msg = "Model not loaded. Call load_model(X, y) first."
            raise RuntimeError(msg)

        X = self.preprocess(inputs)
        X = np.asarray(X)

        # 🔹 Single place where scaling happens at inference
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X

        weights = self._compute_weights(X_scaled)

        preds_per_cluster = [mdl.predict(X_scaled) for mdl in self._cluster_models]
        preds_matrix = np.stack(preds_per_cluster, axis=1)

        blended = np.sum(weights * preds_matrix, axis=1)
        blended = np.asarray(self.postprocess(blended), dtype=np.float64)
        return blended

    @property
    def cluster_centers_(self) -> np.ndarray:
        if self._cluster_centers is None:
            msg = "Model not loaded."
            raise RuntimeError(msg)
        return self._cluster_centers

    @property
    def experts_(self) -> list[Any]:
        if not self.is_loaded():
            msg = "Model not loaded."
            raise RuntimeError(msg)
        return self._cluster_models

    def export_model(
            self,
            save_dir: str,
            filename: str) ->None:
        """
        Export the ensemble model as a pickle file.

        Parameters
        ----------
        save_dir
            Directory where the model will be saved.

        filename
            Name of the file to save the model as.
        """
        if not self.is_loaded():
            msg = "Cannot export an unloaded model."
            raise RuntimeError(msg)

        save_path = Path(save_dir) / f"{filename}.pkl"
        if(not Path(save_dir).exists()):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)

class ClusterWeightedEnsembleSpatialWrapper(BaseModelWrapper):
    """
    Cluster-weighted ensemble of regressors using SKATER (spatially constrained clustering).

    - We run SKATER on gdf[attrs_name] with spatial contiguity from Queen.
    - We train one expert model per SKATER label.
    - At predict time we blend all experts' predictions using exp(-distance)
      to each cluster's mean (in (optionally scaled) feature space).
    """

    def __init__(
        self,
        n_clusters: int,
        model_factory: Callable[[int], Any],
        attrs_name: list[str],
        scaler: StandardScaler | None = None,
        distance_eps: float = 1e-12,
        feature_indices_used_for_clustering: list[int] | None = None,
        decay_constant:float = 2.0
    ) -> None:

        self.n_clusters = n_clusters
        self.model_factory = model_factory
        self.attrs_name = attrs_name
        self.scaler = scaler              # if None → no scaling, if not None → fit inside load_model
        self.distance_eps = distance_eps

        self._skater_labels: np.ndarray | None = None
        self._cluster_models: list[Any] = []
        self._cluster_centers: np.ndarray | None = None
        self._is_loaded: bool = False

        # indices (in attrs_name order) used for clustering / distances
        self.feature_indices_used_for_clustering = feature_indices_used_for_clustering

        self.decay_constant = decay_constant

    def load_model(
        self,
        gdf:gpd.GeoDataFrame|None = None,
        X: np.ndarray|None = None,
        y: np.ndarray|None = None,
    ) -> None:
        """
        Fit the spatial mixture-of-experts:
        1. Run SKATER on (optionally scaled) gdf[attrs_name] to get spatial clusters.
        2. Train one regressor per cluster on X.
        3. Compute per-cluster mean feature vector in the same feature space used for distances.

        Assumptions:
        - gdf, X, y are row-aligned.
        - X is **not** scaled yet; if scaler is provided, it will be fit here.
        - gdf[self.attrs_name] are the same features used to build X (possibly a subset).
        """
        if(gdf is None or X is None or y is None):
            msg = "gdf, X, and y must be provided to load_model."
            raise ValueError(msg)

        X = np.asarray(X)
        y = np.asarray(y)

        # ----- 1. Scaling (optional) -----
        if self.scaler is not None:
            # fit scaler on gdf[attrs_name]
            self.scaler.fit(gdf[self.attrs_name])
            feats_scaled = self.scaler.transform(gdf[self.attrs_name].to_numpy())
            X_scaled = self.scaler.transform(X)
        else:
            feats_scaled = gdf[self.attrs_name].to_numpy()
            X_scaled = X

        # ----- 2. Contiguity + SKATER -----
        w = Queen.from_dataframe(gdf)

        gdf_scaled = gdf.copy()
        gdf_scaled[self.attrs_name] = feats_scaled

        skater_model = Skater(
            gdf=gdf_scaled,
            w=w,
            attrs_name=self.attrs_name,
            n_clusters=self.n_clusters,
        )
        if hasattr(skater_model, "solve"):
            skater_model.solve()

        if not hasattr(skater_model, "labels_"):
            msg = "SKATER returned no labels_. Unexpected version."
            raise RuntimeError(msg)
        labels = np.asarray(skater_model.labels_)

        unique_clusters = np.unique(labels)

        # ----- 3. Prepare feature space for centers / distances -----
        if self.feature_indices_used_for_clustering is not None:
            centers_space = feats_scaled[:, self.feature_indices_used_for_clustering]
        else:
            centers_space = feats_scaled

        cluster_models: list[Any] = []
        cluster_centers_list: list[np.ndarray] = []

        # ----- 4. Train one model per cluster, compute cluster centers -----
        for cid in unique_clusters:
            mask = labels == cid

            model = self.model_factory(int(cid))
            model.fit(X_scaled[mask], y[mask])
            cluster_models.append(model)

            center_k = centers_space[mask].mean(axis=0)
            cluster_centers_list.append(center_k)

        self._cluster_models = cluster_models
        self._cluster_centers = np.vstack(cluster_centers_list)

        # ----- 5. Remap labels to dense 0..n_clusters-1 -----
        remap = {orig_id: new_id for new_id, orig_id in enumerate(unique_clusters)}
        dense_labels = np.array([remap[cid] for cid in labels], dtype=int)
        self._skater_labels = dense_labels

        self._is_loaded = True

    def fit(
        self,
        gdf:gpd.GeoDataFrame,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "ClusterWeightedEnsembleSpatialWrapper":
        self.load_model(gdf, X, y)
        return self

    def is_loaded(self) -> bool:
        return self._is_loaded

    def _compute_weights(self, X_input: np.ndarray) -> np.ndarray:
        """
        Compute soft assignment weights for each sample to each cluster,
        based on distance to cluster centers.

        X_input is in the same space as training X (unscaled); scaling is handled here.
        """
        if self._cluster_centers is None:
            msg = "Model not loaded: no cluster centers."
            raise RuntimeError(msg)

        X_input = np.asarray(X_input)

        # scale if needed, same way as in load_model
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_input)
        else:
            X_scaled = X_input

        if self.feature_indices_used_for_clustering is not None:
            X_for_weights = X_scaled[:, self.feature_indices_used_for_clustering]
        else:
            X_for_weights = X_scaled

        dists = pairwise_distances(X_for_weights, self._cluster_centers)
        dists = np.maximum(dists, self.distance_eps)

        w = np.exp(-self.decay_constant * dists)
        w = w / np.sum(w, axis=1, keepdims=True)
        w = np.asarray(w, dtype = np.float64)
        return w

    def predict(self, inputs: Any) -> np.ndarray:
        """
        Predict heights using as inputs the same space as training X (unscaled).
        Scaling and subspace selection are handled internally.
        """
        if not self.is_loaded():
            msg = "Model not loaded. Call load_model(...) first."
            raise RuntimeError(msg)

        X_input = self.preprocess(inputs)
        X_input = np.asarray(X_input)

        # scale for experts
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_input)
        else:
            X_scaled = X_input

        # weights in the same space centers were defined
        weights = self._compute_weights(X_input)

        preds_per_cluster = [mdl.predict(X_scaled) for mdl in self._cluster_models]
        preds_matrix = np.stack(preds_per_cluster, axis=1)

        blended = np.sum(weights * preds_matrix, axis=1)
        #Convert to array
        blended = np.asarray(self.postprocess(blended), dtype = np.float64)
        return blended

    @property
    def cluster_centers_(self) -> np.ndarray:
        if self._cluster_centers is None:
            msg = "Model not loaded."
            raise RuntimeError(msg)
        return self._cluster_centers

    @property
    def experts_(self) -> list[Any]:
        if not self.is_loaded():
            msg = "Model not loaded."
            raise RuntimeError(msg)
        return self._cluster_models

    @property
    def labels_(self) -> np.ndarray:
        if self._skater_labels is None:
            msg = "Model not loaded."
            raise RuntimeError(msg)
        return self._skater_labels

    def export_model(
            self,
            save_dir: str,
            filename: str) ->None:
        """
        Export the ensemble model as a pickle file.

        Parameters
        ----------
        save_dir
            Directory where the model will be saved.

        filename
            Name of the file to save the model as.
        """
        if not self.is_loaded():
            msg = "Cannot export an unloaded model."
            raise RuntimeError(msg)

        save_path = Path(save_dir) / f"{filename}.pkl"
        if(not Path(save_dir).exists()):
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, save_path)
