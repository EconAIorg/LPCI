"""
Tests confirming that third-party multi-quantile estimators (XGBoost, CatBoost)
work as drop-in replacements for the default RandomForestQuantileRegressor.
"""
import numpy as np
import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
CatBoostRegressor = pytest.importorskip(
    "catboost", reason="catboost not installed"
).CatBoostRegressor


def _run_fit_predict(lpci_instance, estimator):
    """Helper that runs fit_predict with a given estimator and returns interval_df."""
    lpci = lpci_instance
    alpha = 0.1
    n_quantiles = 2
    df, features, target_col = lpci.prepare_df(window_size=1)
    panel_split_kwargs = {"gap": 0, "test_size": 1, "progress_bar": False}

    return lpci.fit_predict(
        df=df,
        features=features,
        target_col=target_col,
        best_params={},
        alpha=alpha,
        n_quantiles=n_quantiles,
        panel_split_kwargs=panel_split_kwargs,
        n_jobs=1,
        estimator=estimator,
    )


def test_fit_predict_xgboost(lpci_instance):
    lpci = lpci_instance
    alpha = 0.1
    n_quantiles = 2
    quantiles = lpci.gen_quantiles(alpha, n_quantiles)

    estimator = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantiles,
        n_estimators=10,
        random_state=0,
    )
    result = _run_fit_predict(lpci_instance, estimator)

    assert "lower_conf" in result.columns
    assert "upper_conf" in result.columns
    assert result["lower_conf"].notna().all()
    assert result["upper_conf"].notna().all()


def test_fit_predict_catboost(lpci_instance):
    lpci = lpci_instance
    alpha = 0.1
    n_quantiles = 2
    quantiles = lpci.gen_quantiles(alpha, n_quantiles)

    alpha_str = ",".join(str(round(q, 5)) for q in quantiles)
    estimator = CatBoostRegressor(
        loss_function=f"MultiQuantile:alpha={alpha_str}",
        iterations=10,
        verbose=0,
    )
    result = _run_fit_predict(lpci_instance, estimator)

    assert "lower_conf" in result.columns
    assert "upper_conf" in result.columns
    assert result["lower_conf"].notna().all()
    assert result["upper_conf"].notna().all()
