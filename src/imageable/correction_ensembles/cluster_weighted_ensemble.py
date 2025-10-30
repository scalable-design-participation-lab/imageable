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

    The idea:
    - Run KMeans to partition feature space into n_clusters.
    - Train one regressor per cluster on that cluster's points.
    - At inference, each new sample is predicted by all cluster models,
      and we take a distance-based soft weighting of those per-cluster predictions.

    This wrapper follows the BaseModelWrapper interface:
    - __init__: configure
    - load_model: "fit"/train on provided (X, y)
    - predict: inference on new X
    - is_loaded: check training status
    - preprocess/postprocess hooks exist and can be overridden
    """

    def __init__(
        self,
        n_clusters: int,
        model_factory: Callable[[int], Any],
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
        distance_eps: float = 1e-12,
    ) -> None:
        """
        Initialize the wrapper with hyperparameters, but do not train.

        Parameters
        ----------
        n_clusters : int
            Number of clusters for KMeans.
        model_factory : Callable[[], Any]
            A callable that returns a fresh regressor with .fit(X, y) and .predict(X).
            This is called once per cluster.
        kmeans_kwargs : dict, optional
            Extra kwargs passed to sklearn.cluster.KMeans.
            e.g. {"n_init": 10, "random_state": 0}
        distance_eps : float
            Small positive constant to avoid zero distance.
        """
        self.n_clusters = n_clusters
        self.model_factory = model_factory
        self.kmeans_kwargs = {} if kmeans_kwargs is None else kmeans_kwargs
        self.distance_eps = distance_eps


        self._kmeans: Optional[KMeans] = None
        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_models: List[Any] = []

        self._is_loaded: bool = False

    def load_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        "Load" the model. In this case, this means: fit the KMeans and the per-cluster regressors.

        Parameters
        ----------
        X : np.ndarray
            Training features, shape (n_samples, n_features)
        y : np.ndarray
            Training targets, shape (n_samples,)
        """
        X = np.asarray(X)
        y = np.asarray(y)


        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            **self.kmeans_kwargs,
        ).fit(X)

        labels = self._kmeans.labels_
        self._cluster_centers = self._kmeans.cluster_centers_

        self._cluster_models = []
        for k in range(self.n_clusters):
            mask = labels == k
            model = self.model_factory(k)
            model.fit(X[mask], y[mask])
            self._cluster_models.append(model)

        self._is_loaded = True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "ClusterWeightedEnsembleWrapper":
        """
        Convenience alias so you can use this like an sklearn estimator if you want.
        """
        self.load_model(X, y)
        return self

    def is_loaded(self) -> bool:
        """
        Returns True if the ensemble has been trained.
        """
        return self._is_loaded

    def _compute_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Compute soft assignment weights for each sample to each cluster,
        based on distance to cluster centers.
        """
        if self._cluster_centers is None:
            raise RuntimeError("Model not loaded: no cluster centers.")
        dists = pairwise_distances(X, self._cluster_centers)
        dists = np.maximum(dists, self.distance_eps)
        w = np.exp(-dists)
        w = w / np.sum(w, axis=1, keepdims=True)
        return w

    def predict(self, inputs: Any) -> np.ndarray:
        """
        Predict target values for new samples.

        Parameters
        ----------
        inputs : Any
            Input feature matrix of shape (n_samples, n_features),
            or any object accepted by preprocess().

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,).
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model(X, y) first.")


        X = self.preprocess(inputs)
        X = np.asarray(X)

        weights = self._compute_weights(X)

        preds_per_cluster = []
        for mdl in self._cluster_models:
            preds_per_cluster.append(mdl.predict(X))
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
      to each cluster's mean (in scaled feature space).
    """

    def __init__(
        self,
        n_clusters: int,
        model_factory: Callable[[int], Any],
        attrs_name: List[str],
        scaler: Optional[StandardScaler] = None,
        distance_eps: float = 1e-12,
    ) -> None:

        self.n_clusters = n_clusters
        self.model_factory = model_factory
        self.attrs_name = attrs_name
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.distance_eps = distance_eps
        self._skater_labels: Optional[np.ndarray] = None
        self._cluster_models: List[Any] = []
        self._cluster_centers: Optional[np.ndarray] = None

        self._is_loaded: bool = False

    def load_model(
        self,
        gdf,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Fit the spatial mixture-of-experts:
        1. SKATER to get spatially constrained clusters.
        2. Train one regressor per cluster.
        3. Compute per-cluster mean feature vector in scaled feature space.

        Assumptions:
        - gdf, X, y are row-aligned.
        - X is ALREADY scaled with the same scaler you passed in __init__.
        - gdf[self.attrs_name] are the SAME features (unscaled) in the SAME order.
        """
        from libpysal.weights import Queen
        from spopt.region import Skater

        X = np.asarray(X)
        y = np.asarray(y)

        w = Queen.from_dataframe(gdf)

        gdf_scaled = gdf.copy()
        gdf_scaled[self.attrs_name] = self.scaler.transform(gdf[self.attrs_name])

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

        self._skater_labels = labels


        unique_clusters = np.unique(labels)


        cluster_models: List[Any] = []
        cluster_centers_list: List[np.ndarray] = []


        X_scaled_full = self.scaler.transform(gdf[self.attrs_name].to_numpy())

        for k in unique_clusters:
            mask = labels == k

            model = self.model_factory(int(k))
            model.fit(X[mask], y[mask])
            cluster_models.append(model)


            center_k = X_scaled_full[mask].mean(axis=0)
            cluster_centers_list.append(center_k)


        self._cluster_models = cluster_models
        self._cluster_centers = np.vstack(cluster_centers_list)


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

    def _compute_weights(self, X: np.ndarray) -> np.ndarray:
        """
        X must already be scaled with the same scaler and in the same column
        order as self.attrs_name.

        We weight each expert by exp(-distance to that expert's cluster center).
        """
        if self._cluster_centers is None:
            raise RuntimeError("Model not loaded: no cluster centers.")

        X_scaled = np.asarray(X)
        dists = pairwise_distances(X_scaled, self._cluster_centers)
        dists = np.maximum(dists, self.distance_eps)

        w = np.exp(-dists)
        w = w / np.sum(w, axis=1, keepdims=True)
        return w

    def predict(self, inputs: Any) -> np.ndarray:
        """
        inputs must already be scaled with the same scaler and same feature order.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model(...) first.")

        X = self.preprocess(inputs)
        X = np.asarray(X)

        weights = self._compute_weights(X)

        preds_per_cluster = []
        for mdl in self._cluster_models:
            preds_per_cluster.append(mdl.predict(X))
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