from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from typing import Tuple, Dict, Callable, List, Optional, Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import pairwise_distances
from imageable.models.base import BaseModelWrapper
from libpysal.weights import Queen  # or Rook
from spopt.region import Skater

def get_best_k_silhouette(
    data: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
) -> Tuple[int, Dict[int, float]]:
    best_k = None
    best_score = -np.inf
    k_min = int(k_range[0])
    k_max = int(k_range[1])
    scores: Dict[int, float] = {}
    data = np.asarray(data, dtype=np.float64)


    if np.isnan(data).any() or np.isinf(data).any():
        raise ValueError("Input data contains NaN or inf.")

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
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
        distance_eps: float = 1e-12,
        feature_indices_used_for_clustering: Optional[List[int]] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.model_factory = model_factory
        self.kmeans_kwargs = {} if kmeans_kwargs is None else kmeans_kwargs
        self.distance_eps = distance_eps

        self._kmeans: Optional[KMeans] = None
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_models: List[Any] = []
        self._is_loaded: bool = False

        # indices of features used only for clustering / distances
        self.feature_indices_used_for_clustering = feature_indices_used_for_clustering

    def load_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit KMeans + per-cluster regressors.

        Parameters
        ----------
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        X_complete = X.copy()

        # features used for clustering (maybe a subset)
        if self.feature_indices_used_for_clustering is not None:
            X_cluster = X[:, self.feature_indices_used_for_clustering]
        else:
            X_cluster = X

        # KMeans on cluster features
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            **self.kmeans_kwargs,
        ).fit(X_cluster)

        labels = self._kmeans.labels_
        self._cluster_centers = self._kmeans.cluster_centers_

        # one expert per cluster, trained on full X
        self._cluster_models = []
        for k in range(self.n_clusters):
            mask = labels == k
            model = self.model_factory(k)
            model.fit(X_complete[mask], y[mask])
            self._cluster_models.append(model)

        self._is_loaded = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClusterWeightedEnsembleWrapper":
        self.load_model(X, y)
        return self

    def is_loaded(self) -> bool:
        return self._is_loaded

    def _compute_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Compute soft assignment weights for each sample to each cluster,
        based on distance to cluster centers.
        """
        if self._cluster_centers is None:
            raise RuntimeError("Model not loaded: no cluster centers.")

        X = np.asarray(X)
        if self.feature_indices_used_for_clustering is not None:
            X = X[:, self.feature_indices_used_for_clustering]

        dists = pairwise_distances(X, self._cluster_centers)
        dists = np.maximum(dists, self.distance_eps)
        w = np.exp(-dists)
        w = w / np.sum(w, axis=1, keepdims=True)
        return w

    def predict(self, inputs: Any) -> np.ndarray:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model(X, y) first.")

        X = self.preprocess(inputs)
        X = np.asarray(X)

        weights = self._compute_weights(X)

        preds_per_cluster = [mdl.predict(X) for mdl in self._cluster_models]
        preds_matrix = np.stack(preds_per_cluster, axis=1)

        blended = np.sum(weights * preds_matrix, axis=1)
        return self.postprocess(blended)

    @property
    def cluster_centers_(self) -> np.ndarray:
        if self._cluster_centers is None:
            raise RuntimeError("Model not loaded.")
        return self._cluster_centers

    @property
    def experts_(self) -> List[Any]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded.")
        return self._cluster_models



import numpy as np
from typing import Any, Callable, Dict, List, Optional

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from libpysal.weights import Queen  # or Rook
from spopt.region import Skater

# assuming you already have this in your codebase
class BaseModelWrapper(BaseEstimator):
    def preprocess(self, X: Any) -> np.ndarray:
        return np.asarray(X)

    def postprocess(self, y_pred: np.ndarray) -> np.ndarray:
        return y_pred

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
        attrs_name: List[str],
        scaler: Optional[StandardScaler] = None,
        distance_eps: float = 1e-12,
        feature_indices_used_for_clustering: Optional[List[int]] = None,
    ) -> None:

        self.n_clusters = n_clusters
        self.model_factory = model_factory
        self.attrs_name = attrs_name
        self.scaler = scaler              # if None → no scaling, if not None → fit inside load_model
        self.distance_eps = distance_eps

        self._skater_labels: Optional[np.ndarray] = None
        self._cluster_models: List[Any] = []
        self._cluster_centers: Optional[np.ndarray] = None
        self._is_loaded: bool = False

        # indices (in attrs_name order) used for clustering / distances
        self.feature_indices_used_for_clustering = feature_indices_used_for_clustering

    def load_model(
        self,
        gdf,
        X: np.ndarray,
        y: np.ndarray,
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
            raise RuntimeError("SKATER returned no labels_. Unexpected version.")
        labels = np.asarray(skater_model.labels_)

        unique_clusters = np.unique(labels)

        # ----- 3. Prepare feature space for centers / distances -----
        if self.feature_indices_used_for_clustering is not None:
            centers_space = feats_scaled[:, self.feature_indices_used_for_clustering]
        else:
            centers_space = feats_scaled

        cluster_models: List[Any] = []
        cluster_centers_list: List[np.ndarray] = []

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
        gdf,
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
            raise RuntimeError("Model not loaded: no cluster centers.")

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

        w = np.exp(-dists)
        w = w / np.sum(w, axis=1, keepdims=True)
        return w

    def predict(self, inputs: Any) -> np.ndarray:
        """
        inputs: same feature space as training X (unscaled).
        Scaling + subspace selection are handled internally.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model(...) first.")

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
        return self.postprocess(blended)

    @property
    def cluster_centers_(self) -> np.ndarray:
        if self._cluster_centers is None:
            raise RuntimeError("Model not loaded.")
        return self._cluster_centers

    @property
    def experts_(self) -> List[Any]:
        if not self.is_loaded():
            raise RuntimeError("Model not loaded.")
        return self._cluster_models

    @property
    def labels_(self) -> np.ndarray:
        if self._skater_labels is None:
            raise RuntimeError("Model not loaded.")
        return self._skater_labels
