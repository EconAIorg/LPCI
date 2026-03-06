# LPCI API Reference

## Custom Estimator Examples

`tune()` and `fit_predict()` both accept an optional `estimator` parameter. Any sklearn-compatible multi-quantile model whose `predict(X)` returns shape `(n_samples, n_quantiles)` can be used as a drop-in replacement for the default `RandomForestQuantileRegressor`.

**Important:** The quantile levels baked into the estimator must match those returned by `gen_quantiles(alpha, n_quantiles)`.

### XGBoost (≥ 2.0)

```python
import xgboost as xgb

alpha = 0.1
n_quantiles = 2
quantiles = lpci.gen_quantiles(alpha, n_quantiles)  # e.g. [0.0, 0.1, 0.9, 1.0]

estimator = xgb.XGBRegressor(
    objective="reg:quantileerror",
    quantile_alpha=quantiles,   # pass the same quantiles array
    n_estimators=100,
    random_state=42,
)

n_splits = lpci.get_n_splits(df[lpci.time_col].unique(), min(lpci.unique_test_time))
cv = PanelSplit(df[lpci.time_col], n_splits=n_splits, gap=0, test_size=1)

interval_df = lpci.fit_predict(
    df=df,
    features=features,
    target_col=target_col,
    best_params={},
    alpha=alpha,
    n_quantiles=n_quantiles,
    cv=cv,
    estimator=estimator,
)
```

To tune XGBoost hyperparameters first:

```python
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV

search = RandomizedSearchCV(
    estimator,
    param_distributions={"n_estimators": randint(50, 300), "max_depth": randint(3, 8)},
    n_iter=10,
    n_jobs=-1,
    cv=3,
)
best_params, quantiles = lpci.tune(
    df=df,
    features=features,
    target_col=target_col,
    alpha=alpha,
    n_quantiles=n_quantiles,
    search=search,
)
```

### CatBoost

```python
from catboost import CatBoostRegressor

alpha = 0.1
n_quantiles = 2
quantiles = lpci.gen_quantiles(alpha, n_quantiles)  # e.g. [0.0, 0.1, 0.9, 1.0]

# CatBoost requires quantile levels as a comma-separated string
alpha_str = ",".join(str(round(q, 5)) for q in quantiles)

estimator = CatBoostRegressor(
    loss_function=f"MultiQuantile:alpha={alpha_str}",
    iterations=100,
    verbose=0,
)

n_splits = lpci.get_n_splits(df[lpci.time_col].unique(), min(lpci.unique_test_time))
cv = PanelSplit(df[lpci.time_col], n_splits=n_splits, gap=0, test_size=1)

interval_df = lpci.fit_predict(
    df=df,
    features=features,
    target_col=target_col,
    best_params={},
    alpha=alpha,
    n_quantiles=n_quantiles,
    cv=cv,
    estimator=estimator,
)
```

---

## `LPCI`

Main class implementing the LPCI algorithm for panel data regression.

```python
from lpci import LPCI
```

---

### `LPCI.__init__`

```python
LPCI(
    eval_delay: int,
    cal_preds: pd.DataFrame,
    test_preds: pd.DataFrame,
    unit_col: str = "unit",
    time_col: str = "year",
    preds_col: str = "preds",
    true_col: str = "true",
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `eval_delay` | `int` | Data delay between the realisation and the time of evaluation (e.g. 1 if data is released one period after observation). |
| `cal_preds` | `pd.DataFrame` | Calibration set — must contain `unit_col`, `time_col`, `preds_col`, `true_col`. |
| `test_preds` | `pd.DataFrame` | Test set — same required columns as `cal_preds`. |
| `unit_col` | `str` | Name of the column identifying cross-sectional units. Default `"unit"`. |
| `time_col` | `str` | Name of the column identifying time periods. Default `"year"`. |
| `preds_col` | `str` | Name of the column containing point predictions. Default `"preds"`. |
| `true_col` | `str` | Name of the column containing true (observed) values. Default `"true"`. |

**Key attributes set on construction:**

| Attribute | Description |
|-----------|-------------|
| `df` | Concatenation of `cal_preds` and `test_preds`. |
| `unique_cal_time` | Sorted array of unique time values in the calibration set. |
| `unique_test_time` | Sorted array of unique time values in the test set. |
| `id_vars` | `[unit_col, time_col]` |

---

### `LPCI.nonconformity_score`

```python
nonconformity_score(df: pd.DataFrame) -> pd.DataFrame
```

Computes residuals (`true - predicted`) and appends them as a `"residuals"` column.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Must contain `unit_col`, `time_col`, `preds_col`, `true_col`. |

**Returns:** `pd.DataFrame` — input DataFrame with an added `"residuals"` column.

---

### `LPCI.lag`

```python
lag(
    df: pd.DataFrame,
    col: str,
    lags: list,
    alpha: float = None,
    adjust: bool = True,
    fillna: Union[float, int] = None,
) -> pd.DataFrame
```

Generates lagged variables for a given column, grouped by `unit_col`. Optionally applies exponential smoothing via `pandas.ewm()`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | DataFrame to generate lagged variables from. |
| `col` | `str` | Column name to lag. |
| `lags` | `list` | List of integer lag periods. |
| `alpha` | `float` | Smoothing factor passed to `pandas.ewm(alpha=...)`. If `None`, no smoothing is applied. See [pandas docs](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html). |
| `adjust` | `bool` | Passed to `pandas.ewm(adjust=...)`. Default `True`. Only used when `alpha` is set. |
| `fillna` | `float` or `int` | Value to fill NaNs in lag columns. If `None`, NaNs are left in place. |

**Returns:** `pd.DataFrame` — input DataFrame with added columns `<col>_lag_<n>` for each lag.

---

### `LPCI.cat_engineer`

```python
cat_engineer(df: pd.DataFrame, cat_method: dict) -> pd.DataFrame
```

Generates features for categorical variables.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | DataFrame to engineer features on. |
| `cat_method` | `dict` | Maps column names to encoding methods. Currently only `"one_hot_encode"` is supported, e.g. `{"isocode": "one_hot_encode"}`. |

**Returns:** `pd.DataFrame` — input DataFrame with one-hot encoded columns added (prefixed `cat_<col>_`).

---

### `LPCI.prepare_df`

```python
prepare_df(
    window_size: int,
    alpha: float = None,
    adjust: bool = True,
    fillna: Union[float, int] = None,
    cat_method: dict = None,
) -> tuple[pd.DataFrame, list, str]
```

Convenience method that chains `nonconformity_score` → `lag` → (optional) `cat_engineer` to produce a model-ready DataFrame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `window_size` | `int` | Number of lagged periods to generate. |
| `alpha` | `float` | Smoothing factor for `pandas.ewm()`. If `None`, no smoothing. |
| `adjust` | `bool` | Passed to `pandas.ewm()`. Default `True`. |
| `fillna` | `float` or `int` | Value to fill NaNs in lag columns. |
| `cat_method` | `dict` | Categorical feature encoding spec (see `cat_engineer`). If `None`, a warning is raised as per the LPCI paper. |

**Returns:** `(df, features, target_col)` where:
- `df` — prepared DataFrame
- `features` — list of feature column names
- `target_col` — `"residuals"`

---

### `LPCI.gen_quantiles`

```python
gen_quantiles(alpha: float, n_quantiles: int) -> np.ndarray
```

Generates an array of quantile levels for the prediction interval.

| Parameter | Type | Description |
|-----------|------|-------------|
| `alpha` | `float` | Significance level (e.g. `0.1` for a 90% interval). |
| `n_quantiles` | `int` | Number of quantiles per side of the interval. |

**Returns:** `np.ndarray` of length `2 * n_quantiles`. The first half are lower quantiles in `[0, alpha]`; the second half are upper quantiles in `[1-alpha, 1]`.

---

### `LPCI.get_n_splits`

```python
get_n_splits(unique_time: list, desired_test_start_time: int) -> int
```

Computes the number of CV splits needed so that the first test fold starts at `desired_test_start_time`. Use this when constructing a `PanelSplit` (or similar) CV object for `fit_predict`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `unique_time` | `list` | Unique time periods in the dataset. |
| `desired_test_start_time` | `int` | Desired start time for the first test fold (typically `min(lpci.unique_test_time)`). |

**Returns:** `int` — number of splits to pass as `n_splits`.

---

### `LPCI.tune`

```python
tune(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    alpha: float,
    n_quantiles: int = 5,
    search=None,
    return_best_estimator: bool = False,
)
```

Tunes hyperparameters for a quantile regression estimator. Training uses only calibration-set observations. The caller provides a fully configured search object — the estimator, CV strategy, and parameter grid are all the user's responsibility.

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Prepared DataFrame from `prepare_df`. |
| `features` | `list` | Feature column names. |
| `target_col` | `str` | Target column name (typically `"residuals"`). |
| `alpha` | `float` | Significance level. |
| `n_quantiles` | `int` | Number of quantiles per side. Default `5`. |
| `search` | sklearn search object | Pre-instantiated search object (e.g. `GridSearchCV`, `RandomizedSearchCV`). The estimator, CV strategy, parameter grid/distributions, and quantile levels must be configured by the caller. |
| `return_best_estimator` | `bool` | If `True`, also return the fitted best estimator. |

**Returns:**
- `best_params_: dict` — best hyperparameters found.
- `quantiles: np.ndarray` — quantile levels used.
- `best_estimator_` *(only if `return_best_estimator=True`)* — fitted estimator.

**Example:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn_quantile import RandomForestQuantileRegressor

quantiles = lpci.gen_quantiles(alpha=0.1, n_quantiles=5)
search = GridSearchCV(
    RandomForestQuantileRegressor(q=quantiles),
    param_grid={"n_estimators": [50, 100, 200]},
    cv=3,
)
best_params, quantiles = lpci.tune(df=df, features=features, target_col=target_col,
                                   alpha=0.1, n_quantiles=5, search=search)
```

---

### `LPCI.gen_conf_interval`

```python
gen_conf_interval(preds: np.ndarray, quantiles: np.ndarray)
```

Constructs prediction intervals from quantile predictions by selecting the narrowest interval.

| Parameter | Type | Description |
|-----------|------|-------------|
| `preds` | `np.ndarray` | Shape `(n_samples, n_quantiles)` — quantile predictions for a single fold. |
| `quantiles` | `np.ndarray` | Quantile levels (from `gen_quantiles`). |

**Returns:** `(lower_conf, upper_conf, opt_lower_q, opt_upper_q)` — per-sample arrays of lower/upper residual bounds and the corresponding optimal quantile levels.

---

### `LPCI.fit_predict`

```python
fit_predict(
    df: pd.DataFrame,
    features: list,
    target_col: str,
    best_params: dict,
    alpha: float,
    n_quantiles: int = 5,
    cv=None,
    n_jobs: int = -1,
    return_fitted_estimators: bool = False,
    estimator=None,
)
```

End-to-end method: fits the quantile estimator via rolling cross-validation and generates prediction intervals on the test set. The caller provides a pre-instantiated CV splitter (e.g. `PanelSplit`, `TimeSeriesSplit`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | Prepared DataFrame from `prepare_df`. |
| `features` | `list` | Feature column names. |
| `target_col` | `str` | Target column name (typically `"residuals"`). |
| `best_params` | `dict` | Hyperparameters to pass to the estimator. |
| `alpha` | `float` | Significance level. |
| `n_quantiles` | `int` | Number of quantiles per side. Default `5`. |
| `cv` | CV splitter | Pre-instantiated cross-validator with a `split(X, y)` method returning positional integer `(train_idx, test_idx)` tuples (sklearn interface). Use `get_n_splits` to compute the correct `n_splits` for `PanelSplit`. |
| `n_jobs` | `int` | Number of parallel jobs for prediction. `-1` uses all CPU cores. `1` runs sequentially. |
| `return_fitted_estimators` | `bool` | If `True`, also return the list of fitted estimators. |
| `estimator` | sklearn estimator | Pre-instantiated multi-quantile estimator. Must be sklearn-compatible and its `predict(X)` must return shape `(n_samples, n_quantiles)`. Quantile levels must match `gen_quantiles(alpha, n_quantiles)`. If `None`, defaults to `RandomForestQuantileRegressor`. |

**Returns:**
- `interval_df: pd.DataFrame` — test-set observations with columns:
  - `q_<value>` — per-quantile residual predictions
  - `<target_col>_lower_conf`, `<target_col>_upper_conf` — residual confidence bounds
  - `opt_lower_q`, `opt_upper_q` — optimal quantile pair used per observation
  - `lower_conf`, `upper_conf` — final prediction interval bounds (point prediction ± residual bound)
- `fitted_estimators: list` *(only if `return_fitted_estimators=True`)* — one fitted estimator per fold.

**Example:**

```python
from panelsplit.cross_validation import PanelSplit

n_splits = lpci.get_n_splits(df[lpci.time_col].unique(), min(lpci.unique_test_time))
cv = PanelSplit(df[lpci.time_col], n_splits=n_splits, gap=0, test_size=1)

interval_df = lpci.fit_predict(
    df=df, features=features, target_col=target_col,
    best_params=best_params, alpha=0.1, n_quantiles=5, cv=cv,
)
```

---

## `EvaluateLPCI`

Evaluation and visualisation utilities for LPCI prediction intervals.

```python
from lpci import EvaluateLPCI
```

---

### `EvaluateLPCI.__init__`

```python
EvaluateLPCI(lpci: LPCI, alpha: float, interval_df: pd.DataFrame)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `lpci` | `LPCI` | A fitted `LPCI` instance (used for column name references). |
| `alpha` | `float` | Significance level of the prediction intervals. |
| `interval_df` | `pd.DataFrame` | Output of `LPCI.fit_predict`. Must contain `unit_col`, `time_col`, `preds_col`, `true_col`, `lower_conf`, `upper_conf`. |

---

### `EvaluateLPCI.overall_coverage`

```python
overall_coverage() -> float
```

Proportion of test observations where the true value falls within `[lower_conf, upper_conf]`.

---

### `EvaluateLPCI.coverage_by_unit`

```python
coverage_by_unit() -> pd.Series
```

Coverage grouped by unit. Index is the unit column values.

---

### `EvaluateLPCI.coverage_by_time`

```python
coverage_by_time() -> pd.Series
```

Coverage grouped by time period. Index is the time column values.

---

### `EvaluateLPCI.coverage_by_bin`

```python
coverage_by_bin(bins: list, bin_labels: list) -> pd.Series
```

Coverage grouped by bins of the true value.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bins` | `list` | Bin edges (e.g. `[0, 100, 1000, float('inf')]`). |
| `bin_labels` | `list` | Labels for each bin (length `len(bins) - 1`). |

**Returns:** `pd.Series` indexed by bin label.

---

### `EvaluateLPCI.plot_intervals_year`

```python
plot_intervals_year(year: int) -> matplotlib.figure.Figure
```

Plots prediction intervals for all units in a given year.

| Parameter | Type | Description |
|-----------|------|-------------|
| `year` | `int` | The time period to plot. |

---

### `EvaluateLPCI.plot_intervals_unit`

```python
plot_intervals_unit(unit: str) -> matplotlib.figure.Figure
```

Plots prediction intervals for all time periods for a given unit.

| Parameter | Type | Description |
|-----------|------|-------------|
| `unit` | `str` | The unit identifier to plot. |
