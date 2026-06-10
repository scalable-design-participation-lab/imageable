from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from imageable._correction_ensembles.model_factories import (
    choose_model_for_cluster,
    choose_model_for_cluster_gb,
    choose_model_for_cluster_xgb,
)


def test_choose_model_for_cluster():
    k = 12
    model = choose_model_for_cluster(k)
    #Assert that the type is XGBoost
    assert isinstance(model, XGBRegressor)


def test_choose_model_for_cluster_gb():
    k = 12
    model = choose_model_for_cluster_gb(k)
    assert isinstance(model, GradientBoostingRegressor)


def test_xgb_choose_model_for_cluster():
    k = 12
    model = choose_model_for_cluster_xgb(k)
    assert isinstance(model, XGBRegressor)







