"""
This module implements the LPCI algorithm for regression forecasting models.
"""

import warnings
from typing import Union

import pandas as pd
import numpy as np
from sklearn.base import clone
import multiprocessing as mp
import os

class LPCI:
    """
    Class that implements the LPCI algorithm for a regression forecasting model.

    Args
    ----

    eval_delay: int
        Data delay between the realisation and the time of evaluation.

    cal_preds: pd.DataFrame
        DataFrame containing the predictions of the model on the calibration set.
        It must contain the columns provided upon initialization:
        [unit_col, time_col, preds_col, true_col]
        The time_col should represent the time of the prediction, not the time of the true value.

    test_preds: pd.DataFrame
        DataFrame containing the predictions of the model on the test set.
        It must contain the columns provided upon initialization:
        [unit_col, time_col, preds_col, true_col]
        The time_col should represent the time of the prediction, not the time of the true value.

    unit_col: str
        Name of the column containing the unit identifier.

    time_col: str
        Name of the column containing the time identifier.

    preds_col: str
        Name of the column containing the point predictions of the model.

    true_col: str
        Name of the column containing the true values.

    Attributes
    ----

    eval_delay: int
        Data delay between the realisation and the time of evaluation.
    
    unit_col: str
        Name of the column containing the unit identifier.

    time_col: str
        Name of the column containing the time identifier.

    id_vars: list
        List of columns to use as identifiers.
        [unit_col, time_col]
    
    preds_col: str
        Name of the column containing the point predictions of the model.
    
    true_col: str
        Name of the column containing the true values.

    cal_preds: pd.DataFrame
        DataFrame containing the predictions of the model on the calibration set.
    
    test_preds: pd.DataFrame
        DataFrame containing the predictions of the model on the test set.
    
    df: pd.DataFrame
        DataFrame containing the predictions of the model on the calibration and test sets.

    unique_cal_time: list
        List of unique time periods in the calibration set.
    
    unique_test_time: list
        List of unique time periods in the test set.
    """

    def __init__(
        self,
        eval_delay: int,
        cal_preds: pd.DataFrame,
        test_preds: pd.DataFrame,
        unit_col: str = "unit",
        time_col: str = "year",
        preds_col: str = "preds",
        true_col: str = "true",
    ):

        self.eval_delay = eval_delay
        self.unit_col = unit_col
        self.time_col = time_col
        self.id_vars = [unit_col, time_col]
        self.preds_col = preds_col
        self.true_col = true_col

        # DataFrames must be sorted by [unit_col, time_col]
        self.cal_preds = cal_preds.sort_values(by=self.id_vars).reset_index(drop=True)
        self.test_preds = test_preds.sort_values(by=self.id_vars).reset_index(drop=True)
        self.df = (
            pd.concat([self.cal_preds, self.test_preds], axis=0)
            .sort_values(by=self.id_vars)
            .reset_index(drop=True)
        )

        # check that the time_col is of type int
        self._dtype_check(self.df, self.time_col, np.dtype("int64"))

        # obtain the unique time periods in the calibration set
        self.unique_cal_time = sorted(self.cal_preds[self.time_col].unique())
        # obtain the unique time periods in the test set
        self.unique_test_time = sorted(self.test_preds[self.time_col].unique())

    def _dtype_check(self, df: pd.DataFrame, col: str, dtype):
        """
        Method that checks if a column in a DataFrame has the specified data type.

        Args
        ------

        df: pd.DataFrame
            DataFrame to check.

        col: str
            Name of the column to check.

        dtype: dtype
            Data type to check.
        """

        if df[col].dtype != dtype:
            raise ValueError(
                f"The column {col} must be of type {dtype}. Currently, the type is {df[col].dtype}."
            )

    def get_n_splits(self, unique_time: list, desired_test_start_time: int) -> int:
        """
        Computes the number of CV splits needed so that the first test fold starts at
        ``desired_test_start_time``.  Pass this value as ``n_splits`` when constructing
        a ``PanelSplit`` (or similar) CV object for use with :meth:`fit_predict`.

        Args
        ----
        unique_time : list
            The unique time periods in the dataset.
        desired_test_start_time : int
            The desired start time for the first test fold.

        Returns
        -------
        n_splits : int
            The number of splits to be used in the rolling forecast.
        """

        unique_time = sorted(unique_time)
        n_splits = len(unique_time[unique_time.index(desired_test_start_time) :])

        return n_splits

    def _predict_split(
        self, fitted_estimator, X_test: pd.DataFrame
    ):
        """
        Method that generates the predictions of the quantile regression forest.

        Args
        ------

        fitted_estimator: RandomForestQuantileRegressor
            Fitted quantile regression forest.

        X_test: pd.DataFrame
            DataFrame containing the features for the test set.

        Returns
        ------
        np.array (n_samples, n_quantiles)
            Array containing the predictions of the quantile regression forest.

        pd.Index
            Index of the test set.
        """

        preds = fitted_estimator.predict(X_test)
        # sklearn_quantile returns (n_quantiles, n_samples); normalize to (n_samples, n_quantiles)
        if preds.ndim == 2 and preds.shape[0] != len(X_test):
            preds = preds.T

        return preds, X_test.index

    @staticmethod
    def predict_split_mp(args):
        """
        Helper method for multiprocessing.

        Args
        ------

        args: tuple
            Tuple containing the fitted estimator, X_test, and the split index.
        
        Returns
        ------
        
        np.array (n_samples, n_quantiles)
            Array containing the predictions of the quantile regression forest.
        
        pd.Index
            Index of the test set.
        """
        estimator, X_test, split_index = args
        preds = estimator.predict(X_test)
        # sklearn_quantile returns (n_quantiles, n_samples); normalize to (n_samples, n_quantiles)
        if preds.ndim == 2 and preds.shape[0] != len(X_test):
            preds = preds.T
        return preds, X_test.index

    def nonconformity_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that computes the nonconformity score for a given dataset.
        In the context of regression, the nonconformity score is simply the residual (true - predicted).

        Parameters
        ----

        df: pd.DataFrame
            DataFrame containing the predictions of the model.
            It must contain the columns provided upon initialization: [unit_col, time_col, preds_col, true_col]
            The time_col should represent the time of the prediction, not the time of the true values.

        Returns
        ----

        pd.DataFrame
            DataFrame containing the residuals of the model.
        """

        df = df.copy()
        df["residuals"] = df[self.true_col] - df[self.preds_col]

        return df

    def lag(
        self,
        df: pd.DataFrame,
        col: str,
        lags: list,
        alpha: float = None,
        adjust: bool = True,
        fillna: Union[float, int] = None,
    ) -> pd.DataFrame:
        """
        Method that generates lagged variables for a given DataFrame.

        Parameters
        ----

        df: pd.DataFrame
            DataFrame to generate the lagged variables.

        col: str
            Name of the column to generate the lags.

        lags: list
            List of integers with the lags to generate.

        alpha: float
            The alpha (smoothing factor) passed to pandas.ewm(alpha=...) for exponential smoothing
            of the lagged values. If None, no smoothing is applied. See pandas.DataFrame.ewm() for details.

        adjust: bool
            Whether to adjust the weights for the exponential moving average.
            Only used if alpha is not None.
            Pandas default is True. When adjust = False, weighted averages are calculated recursively.

        fillna: Union[float, int]
            Value to fill NaNs in the residuals.

        Returns
        ----

        pd.DataFrame
            DataFrame with the lagged variables.
        """

        df = df.copy()
        for lag in lags:

            #generate the lagged variables
            df[f'{col}_lag_{lag}'] = df.groupby(self.unit_col, observed = True)[col].shift(lag + self.eval_delay)

            #smooth the residuals if alpha is not None
            if alpha is not None:
                df[f'{col}_lag_{lag}'] = df.groupby(self.unit_col, observed = True)[f'{col}_lag_{lag}'].transform(lambda x: x.ewm(alpha = alpha, adjust = adjust).mean())

            #fill NaNs if fillna is not None
            if fillna is not None:
                df[f'{col}_lag_{lag}'] = df[f'{col}_lag_{lag}'].fillna(fillna)
                
        return df

    def cat_engineer(self, df: pd.DataFrame, cat_method: dict) -> pd.DataFrame:
        """
        Method that generates features for a categorical variable.

        Parameters
        ----

        df: pd.DataFrame
            DataFrame to generate the categorical feature on.

        cat_method: dict
            Dictionary containing the method to apply to the categorical feature.
            The keys are the column names and the values are the method to apply
            Currently, the only supported method is 'one_hot_encode'.

        Returns
        ----

        pd.DataFrame
            DataFrame with the original columns + categorical feature.
        """

        for col, method in cat_method.items():

            self._dtype_check(df, col, "category")

            if method != "one_hot_encode":
                raise NotImplementedError(
                    'Currently, the only supported cat_method is "one_hot_encode".'
                )

            dummies = pd.get_dummies(
                df[col], prefix=f"cat_{col}", drop_first=True, dtype=int
            )
            df = pd.concat([df, dummies], axis=1)

        return df

    def prepare_df(
        self,
        window_size: int,
        alpha: float = None,
        adjust: bool = True,
        fillna: Union[float, int] = None,
        cat_method: dict = None,
    ):
        """
        Method that generates X_train and y_train for the quantile regression forest.

        Args
        ------

        window_size:int
            Size of the window to generate the lagged variables.

        alpha: float
            The alpha (smoothing factor) passed to pandas.ewm(alpha=...) for exponential smoothing
            of the lagged residuals. If None, no smoothing is applied. See pandas.DataFrame.ewm() for details.

        fillna: Union[float, int]
            Value to fill NaNs in the residuals.

        adjust: bool
            Whether to adjust the weights for the exponential moving average.
            Pandas default is True. When adjust = False, weighted averages are calculated recursively.

        cat_method: dict
            Dictionary containing the method to apply to the categorical feature.
            The keys are the column names and the values are the method to apply.
            Currently, the only supported method is 'one_hot_encode'.        

        Returns
        ------

        df: pd.DataFrame
            DataFrame containing the features and true values.

        features: list
            List of features to include in the model.

        target_col: str
            Name of the target variable.
        """

        # first generate nonconformity scores (residuals)
        df = self.nonconformity_score(self.df)

        # now generate the lagged residuals
        lags = np.arange(1, window_size + 1)

        df = self.lag(
            df=df, 
            col="residuals", 
            lags=lags, 
            alpha=alpha,
            adjust=adjust,
            fillna=fillna
            )

        # drop the rows with NaNs - these are observations where we cannot generate lagged variables for a given window.
        df = df.dropna(subset=[x for x in df.columns if "lag" in x], axis=0, how="any")

        # collect columns
        features = [x for x in df.columns if "lag" in x]
        target_col = "residuals"

        # now generate dummy variables for the units if group_identifier is specified
        if cat_method is None:
            warnings.warn(
                "The official LPCI algorithm as per Batra et al (2023) states that a group identifier is required. By not including group identifiers, you will be implementing a method closer to the SPCI algorithm as per Xu and Xie (2022)."
            )

        else:
            df = self.cat_engineer(df, cat_method)
            features += [x for x in df.columns if "cat" in x]

        return df, features, target_col

    def gen_quantiles(self, alpha: float, n_quantiles: int):
        """
        Method that generates the quantiles for the quantile regression forest.

        Args
        ------

        alpha: float
            Significance level for the prediction interval.

        n_quantiles: int
            Number of quantiles to generate for either side of the prediction interval.

        Returns
        ------

        np.array
            Array containing the quantiles to pass to the quantile regression forest.
        """

        lower_quantiles = np.linspace(start=0, stop=alpha, num=n_quantiles)
        upper_quantiles = 1 - alpha + lower_quantiles

        return np.concatenate([lower_quantiles, upper_quantiles])

    def tune(
        self,
        df: pd.DataFrame,
        features: list,
        target_col: str,
        alpha: float,
        n_quantiles: int = 5,
        search=None,
        return_best_estimator: bool = False,
    ):
        """
        Tunes hyperparameters for a quantile regression estimator.

        Training is restricted to calibration-set time periods. The caller is
        responsible for constructing a fully configured search object (e.g.
        ``GridSearchCV`` or ``RandomizedSearchCV``) with the estimator, CV
        strategy, and parameter grid already baked in.

        Args
        ------

        df: pd.DataFrame
            DataFrame from :meth:`prepare_df`.

        features: list
            Feature column names.

        target_col: str
            Target column name (typically ``"residuals"``).

        alpha: float
            Significance level for the prediction interval.

        n_quantiles: int
            Number of quantiles per side of the interval. Default ``5``.

        search: sklearn-compatible search object
            A pre-instantiated search object such as ``GridSearchCV`` or
            ``RandomizedSearchCV``.  The estimator, CV strategy, parameter
            grid/distributions, and quantile levels must all be configured by
            the caller before passing here.

        return_best_estimator: bool
            If ``True``, also return the fitted best estimator.

        Returns
        ------
        best_params_: dict
            Best hyperparameters found by the search.

        quantiles: np.ndarray
            Quantile levels used (from :meth:`gen_quantiles`).

        best_estimator_: estimator (only if ``return_best_estimator=True``)
            The fitted best estimator from ``search``.
        """

        self._dtype_check(df, self.time_col, np.dtype("int64"))

        train_df = (
            df[df[self.time_col].isin(self.unique_cal_time)]
            .sort_values(by=self.id_vars)
            .reset_index(drop=True)
        )

        X_train = train_df[features]
        y_train = train_df[target_col]

        quantiles = self.gen_quantiles(alpha, n_quantiles)

        search.fit(X_train, y_train)

        if return_best_estimator:
            return search.best_params_, quantiles, search.best_estimator_
        return search.best_params_, quantiles

    def gen_conf_interval(self, preds: np.array, quantiles: np.array):
        """
        Method that unpacks the predictions of the quantile regression forest.

        Args
        ------

        preds: np.array (n_samples, n_quantiles)
            Array containing the predictions of the quantile regression forest.

        quantiles: np.array
            Array containing the quantiles for the prediction.

        Returns
        ------

        lower_conf: np.array
            Array containing the lower confidence interval.

        upper_conf: np.array
            Array containing the upper confidence interval.

        optimal_quantiles: list
            List of tuples containing the optimal quantiles for the prediction interval.

        """

        # split the quantiles into lower and upper
        mid_quantile = int(len(quantiles) / 2)
        lower_quantile_preds = preds[:, :mid_quantile]
        upper_quantile_preds = preds[:, mid_quantile:]

        # calculate the width of the prediction interval
        widths = upper_quantile_preds - lower_quantile_preds

        # we want to find the narrowest prediction interval that guarantees the coverage of the prediction interval
        i_stars = np.argmin(
            widths, axis=1
        )  # get the index that corresponds to the smallest width. shape (n_samples,)

        # get the lower and upper confidence intervals
        lower_conf = lower_quantile_preds[np.arange(len(i_stars)), i_stars]
        upper_conf = upper_quantile_preds[np.arange(len(i_stars)), i_stars]

        # get the optimal quantiles
        opt_lower_q = quantiles[:mid_quantile][i_stars]
        opt_upper_q = quantiles[mid_quantile:][i_stars]

        return lower_conf, upper_conf, opt_lower_q, opt_upper_q

    def fit_predict(
        self,
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
    ):
        """
        Fits the quantile estimator via rolling CV and generates prediction intervals
        on the test set.

        Args
        ------

        df: pd.DataFrame
            DataFrame from :meth:`prepare_df`.

        features: list
            Feature column names.

        target_col: str
            Target column name (typically ``"residuals"``).

        best_params: dict
            Hyperparameters passed to the estimator via ``set_params``.

        alpha: float
            Significance level for the prediction interval.

        n_quantiles: int
            Number of quantiles per side of the interval. Default ``5``.

        cv: CV splitter
            A pre-instantiated cross-validator with a ``split(X, y)`` method
            returning ``(train_idx, test_idx)`` as positional integer indices
            (sklearn interface).  Any sklearn-compatible splitter works, e.g.
            ``PanelSplit``, ``TimeSeriesSplit``, etc.
            Use :meth:`get_n_splits` to compute the correct ``n_splits`` when
            constructing a ``PanelSplit``.

        n_jobs: int
            Number of parallel prediction jobs. ``-1`` uses all CPU cores,
            ``1`` runs sequentially.

        return_fitted_estimators: bool
            If ``True``, also return the list of fitted estimators.

        estimator: sklearn-compatible estimator, optional
            A pre-instantiated multi-quantile estimator. Must be
            sklearn-compatible and ``predict(X)`` must return shape
            ``(n_samples, n_quantiles)``. Quantile levels must match
            ``gen_quantiles(alpha, n_quantiles)``.
            If ``None``, defaults to ``RandomForestQuantileRegressor``.

        Returns
        ------
        interval_df: pd.DataFrame
            Test-set observations with added columns: per-quantile predictions
            (``q_<value>``), residual confidence bounds
            (``<target_col>_lower_conf``, ``<target_col>_upper_conf``),
            optimal quantile pairs (``opt_lower_q``, ``opt_upper_q``), and
            final interval bounds (``lower_conf``, ``upper_conf``).

        fitted_estimators: list (only if ``return_fitted_estimators=True``)
            One fitted estimator per CV fold.
        """

        self._dtype_check(df, self.time_col, np.dtype("int64"))

        quantiles = self.gen_quantiles(alpha, n_quantiles)

        # Materialise splits once so we can iterate multiple times
        splits = list(cv.split(df[features], df[target_col]))

        # Build interval_df from the test folds
        test_dfs = [df.iloc[test_idx][self.id_vars + [target_col]] for _, test_idx in splits]
        interval_df = pd.concat(test_dfs)
        interval_df = interval_df.reset_index()  # preserve original label index
        interval_df = interval_df.merge(
            self.test_preds[self.id_vars + [self.preds_col, self.true_col]],
            on=self.id_vars,
            how="left",
        )
        interval_df = interval_df.set_index("index")
        interval_df.index.name = None

        # Initialise the quantile regressor
        if estimator is None:
            from sklearn_quantile import RandomForestQuantileRegressor
            qrf = RandomForestQuantileRegressor(q=quantiles, **best_params)
        else:
            qrf = clone(estimator)
            if best_params:
                qrf.set_params(**best_params)

        # Fit one estimator per fold
        fitted_estimators = []
        for train_idx, _ in splits:
            est = clone(qrf)
            est.fit(df.iloc[train_idx][features], df.iloc[train_idx][target_col])
            fitted_estimators.append(est)

        # Predict
        if n_jobs == 1:
            test_preds_list = []
            for i, (_, test_idx) in enumerate(splits):
                preds, index = self._predict_split(
                    fitted_estimators[i], df.iloc[test_idx][features]
                )
                test_preds_list.append((preds, index))
        else:
            if n_jobs <= -1:
                num_processes = os.cpu_count()
            elif n_jobs > 0:
                num_processes = n_jobs
            else:
                raise ValueError(f"Invalid n_jobs value: {n_jobs}")

            args = [
                (fitted_estimators[i], df.iloc[test_idx][features].copy(), i)
                for i, (_, test_idx) in enumerate(splits)
            ]
            with mp.Pool(processes=num_processes) as pool:
                test_preds_list = pool.map(LPCI.predict_split_mp, args)

        # Unpack predictions into interval_df
        for preds, index in test_preds_list:
            interval_df.loc[index, [f"q_{np.round(q, 5)}" for q in quantiles]] = preds
            lower_conf, upper_conf, opt_lower_q, opt_upper_q = self.gen_conf_interval(
                preds, quantiles
            )
            interval_df.loc[index, f"{target_col}_lower_conf"] = lower_conf
            interval_df.loc[index, f"{target_col}_upper_conf"] = upper_conf
            interval_df.loc[index, "opt_lower_q"] = opt_lower_q
            interval_df.loc[index, "opt_upper_q"] = opt_upper_q

        interval_df["lower_conf"] = (
            interval_df[self.preds_col] + interval_df[f"{target_col}_lower_conf"]
        )
        interval_df["upper_conf"] = (
            interval_df[self.preds_col] + interval_df[f"{target_col}_upper_conf"]
        )

        if return_fitted_estimators:
            return interval_df, fitted_estimators
        return interval_df
    
    