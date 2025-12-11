from sklearn.ensemble import GradientBoostingRegressor

def choose_model_for_cluster(k: int):
    """
    Factory function for creating a regression model for a given cluster.

    IMPORTANT:
    This function MUST remain importable at module level so that
    pickled ClusterWeightedEnsembleWrapper objects can be loaded
    without errors.

    You may later extend this to assign different models per cluster.
    """
    return GradientBoostingRegressor(random_state=0)

