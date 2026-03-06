import numpy as np
import pytest
from sklearn_quantile import RandomForestQuantileRegressor

from lpci import LPCI


def test_init(lpci_instance):
    lpci = lpci_instance
    assert lpci.unit_col == "unit"
    assert lpci.time_col == "year"
    assert lpci.eval_delay == 1
    assert len(lpci.df) == len(lpci.cal_preds) + len(lpci.test_preds)


def test_nonconformity_score(lpci_instance):
    lpci = lpci_instance
    df = lpci.nonconformity_score(lpci.df)
    assert "residuals" in df.columns
    expected = lpci.df[lpci.true_col] - lpci.df[lpci.preds_col]
    assert (df["residuals"] == expected).all()


def test_lag_basic(lpci_instance):
    lpci = lpci_instance
    df = lpci.nonconformity_score(lpci.df)
    lags = [1, 2]
    result = lpci.lag(df, "residuals", lags)
    for lag in lags:
        assert f"residuals_lag_{lag}" in result.columns


def test_lag_alpha(lpci_instance):
    lpci = lpci_instance
    df = lpci.nonconformity_score(lpci.df)
    lags = [1]
    plain = lpci.lag(df, "residuals", lags)
    smoothed = lpci.lag(df, "residuals", lags, alpha=0.5)
    # Smoothed values should generally differ from plain lags
    col = "residuals_lag_1"
    diff = (plain[col].dropna() != smoothed[col].dropna()).any()
    assert diff


def test_lag_fillna(lpci_instance):
    lpci = lpci_instance
    df = lpci.nonconformity_score(lpci.df)
    result = lpci.lag(df, "residuals", [1, 2], fillna=0)
    lag_cols = [c for c in result.columns if "residuals_lag_" in c]
    assert result[lag_cols].isna().sum().sum() == 0


def test_cat_engineer(lpci_instance):
    lpci = lpci_instance
    df = lpci.nonconformity_score(lpci.df)
    result = lpci.cat_engineer(df, {"unit": "one_hot_encode"})
    dummy_cols = [c for c in result.columns if c.startswith("cat_unit_")]
    assert len(dummy_cols) > 0


def test_prepare_df(lpci_instance):
    lpci = lpci_instance
    df, features, target_col = lpci.prepare_df(window_size=1)
    assert isinstance(features, list)
    assert len(features) > 0
    assert target_col == "residuals"
    assert "residuals" in df.columns


def test_gen_quantiles(lpci_instance):
    lpci = lpci_instance
    alpha = 0.1
    n_quantiles = 3
    quantiles = lpci.gen_quantiles(alpha, n_quantiles)
    assert len(quantiles) == 2 * n_quantiles
    assert all(0 <= q <= 1 for q in quantiles)
    # Lower half should all be <= alpha, upper half should all be >= 1-alpha
    assert all(q <= alpha for q in quantiles[:n_quantiles])
    assert all(q >= 1 - alpha for q in quantiles[n_quantiles:])


def test_fit_predict_returns_df(lpci_instance, fitted_interval_df):
    lpci = lpci_instance
    interval_df = fitted_interval_df
    assert set(lpci.unique_test_time).issubset(set(interval_df[lpci.time_col].unique()))
    assert "lower_conf" in interval_df.columns
    assert "upper_conf" in interval_df.columns


def test_fit_predict_custom_estimator(lpci_instance):
    from panelsplit.cross_validation import PanelSplit

    lpci = lpci_instance
    window_size = 1
    df, features, target_col = lpci.prepare_df(window_size=window_size)

    alpha = 0.1
    n_quantiles = 2
    quantiles = lpci.gen_quantiles(alpha, n_quantiles)
    custom_estimator = RandomForestQuantileRegressor(q=quantiles, n_estimators=10, random_state=0)

    n_splits = lpci.get_n_splits(df[lpci.time_col].unique(), min(lpci.unique_test_time))
    cv = PanelSplit(df[lpci.time_col], n_splits=n_splits, gap=0, test_size=1, progress_bar=False)

    result = lpci.fit_predict(
        df=df,
        features=features,
        target_col=target_col,
        best_params={},
        alpha=alpha,
        n_quantiles=n_quantiles,
        cv=cv,
        n_jobs=1,
        estimator=custom_estimator,
    )
    assert "lower_conf" in result.columns
    assert "upper_conf" in result.columns
