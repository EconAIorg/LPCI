import itertools
import numpy as np
import pandas as pd
import pytest
from lpci import LPCI, EvaluateLPCI


@pytest.fixture(scope="session")
def lpci_instance():
    np.random.seed(42)
    units = ["A", "B", "C"]
    cal_years = np.arange(2015, 2020, dtype=int)
    test_years = np.arange(2020, 2022, dtype=int)
    eval_delay = 1

    def _make_set(years):
        df = pd.DataFrame(
            [[u, y] for u, y in itertools.product(units, years)],
            columns=["unit", "year"],
        )
        df["unit"] = df["unit"].astype("category")
        df["year"] = df["year"].astype(int)
        df["preds"] = np.abs(np.random.normal(0, 1000, len(df)))
        df["true"] = np.abs(np.random.normal(0, 1000, len(df)))
        return df

    cal_set = _make_set(cal_years)
    test_set = _make_set(test_years)

    return LPCI(
        eval_delay=eval_delay,
        cal_preds=cal_set,
        test_preds=test_set,
        unit_col="unit",
        time_col="year",
        preds_col="preds",
        true_col="true",
    )


@pytest.fixture(scope="session")
def fitted_interval_df(lpci_instance):
    from panelsplit import PanelSplit

    lpci = lpci_instance
    window_size = 1
    df, features, target_col = lpci.prepare_df(window_size=window_size)

    alpha = 0.1
    n_quantiles = 2
    best_params = {"n_estimators": 10, "random_state": 0}

    n_splits = lpci.get_n_splits(df[lpci.time_col].unique(), min(lpci.unique_test_time))
    cv = PanelSplit(df[lpci.time_col], n_splits=n_splits, gap=0, test_size=1, progress_bar=False)

    return lpci.fit_predict(
        df=df,
        features=features,
        target_col=target_col,
        best_params=best_params,
        alpha=alpha,
        n_quantiles=n_quantiles,
        cv=cv,
        n_jobs=1,
    )
