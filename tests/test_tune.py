"""
Tests confirming that tune() works with pre-instantiated sklearn search objects.
"""
import numpy as np
import pytest
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn_quantile import RandomForestQuantileRegressor


def test_tune_gridsearch_kfold(lpci_instance):
    lpci = lpci_instance
    alpha = 0.1
    n_quantiles = 2
    quantiles = lpci.gen_quantiles(alpha, n_quantiles)

    df, features, target_col = lpci.prepare_df(window_size=1)

    estimator = RandomForestQuantileRegressor(q=quantiles)
    search = GridSearchCV(
        estimator,
        param_grid={"n_estimators": [5, 10]},
        cv=3,
    )

    best_params, returned_quantiles = lpci.tune(
        df=df,
        features=features,
        target_col=target_col,
        alpha=alpha,
        n_quantiles=n_quantiles,
        search=search,
    )

    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert len(returned_quantiles) == 2 * n_quantiles


def test_tune_randomized_panelsplit(lpci_instance):
    from panelsplit.cross_validation import PanelSplit
    from scipy.stats import randint

    lpci = lpci_instance
    alpha = 0.1
    n_quantiles = 2
    quantiles = lpci.gen_quantiles(alpha, n_quantiles)

    df, features, target_col = lpci.prepare_df(window_size=1)

    # Build a PanelSplit over calibration data only.
    # n_splits must be < number of unique cal periods in the prepared df (TimeSeriesSplit constraint).
    cal_df = df[df[lpci.time_col].isin(lpci.unique_cal_time)]
    n_unique_cal = len(cal_df[lpci.time_col].unique())
    n_splits = max(2, n_unique_cal - 1)
    cv = PanelSplit(cal_df[lpci.time_col], n_splits=n_splits, gap=0, test_size=1, progress_bar=False)

    estimator = RandomForestQuantileRegressor(q=quantiles)
    search = RandomizedSearchCV(
        estimator,
        param_distributions={"n_estimators": randint(5, 15)},
        n_iter=3,
        cv=cv,
    )

    best_params, returned_quantiles = lpci.tune(
        df=df,
        features=features,
        target_col=target_col,
        alpha=alpha,
        n_quantiles=n_quantiles,
        search=search,
    )

    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert len(returned_quantiles) == 2 * n_quantiles


def test_tune_return_best_estimator(lpci_instance):
    lpci = lpci_instance
    alpha = 0.1
    n_quantiles = 2
    quantiles = lpci.gen_quantiles(alpha, n_quantiles)

    df, features, target_col = lpci.prepare_df(window_size=1)

    estimator = RandomForestQuantileRegressor(q=quantiles)
    search = GridSearchCV(
        estimator,
        param_grid={"n_estimators": [5, 10]},
        cv=3,
    )

    best_params, returned_quantiles, best_estimator = lpci.tune(
        df=df,
        features=features,
        target_col=target_col,
        alpha=alpha,
        n_quantiles=n_quantiles,
        search=search,
        return_best_estimator=True,
    )

    assert isinstance(best_params, dict)
    assert hasattr(best_estimator, "predict")
