from sklearn.ensemble import GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


def choose_model_for_cluster_gb(k: int):
    """Legacy/fallback factory based on GradientBoostingRegressor."""
    return GradientBoostingRegressor(random_state=0)


def choose_model_for_cluster_xgb(k: int):
    """
    XGBoost-first factory for cluster experts.

    Falls back to GradientBoostingRegressor if xgboost is unavailable, so
    this function remains safe to import in lightweight environments.
    """
    if XGBRegressor is None:
        return choose_model_for_cluster_gb(k)

    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=-1,
    )


def choose_model_for_cluster(k: int):
    """
    Default factory used by training code and loaded pickle artifacts.

    IMPORTANT:
    This function MUST remain importable at module level so that
    pickled ClusterWeightedEnsembleWrapper objects can be loaded
    without errors.
    """
    return choose_model_for_cluster_xgb(k)
