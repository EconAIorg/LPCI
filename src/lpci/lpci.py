"""
This module implements the LPCI algorithm for regression forecasting models.
It supports the use of a RandomForestQuantileRegressor for generating predictions
and confidence intervals.
"""

import warnings
from typing import Union

import pandas as pd
import numpy as np
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from panelsplit import PanelSplit
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp
import os

class LPCI:
    """
    Class that implements the LPCI algorithm for a regression forecasting model.
    Currently only the use of a RandomForestQuantileRegressor is supported.

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

    def _get_n_splits(self, unique_time: list, desired_test_start_time: int) -> int:
        """
        This function is used to split the data into training and test sets based on the unique
        time in the dataset.

        Args
        ----
        unique_time : list
            The unique time in the dataset.
        desired_test_start_time : int
            The desired start time for the test set.

        Returns
        -------
        n_splits : int
            The number of splits to be used in the rolling forecast.
        """

        unique_time = sorted(unique_time)
        n_splits = len(unique_time[unique_time.index(desired_test_start_time) :])

        return n_splits

    def _predict_split(
        self, fitted_estimator: RandomForestQuantileRegressor, X_test: pd.DataFrame
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

        # shape (n_quantiles, n_samples)
        preds = fitted_estimator.predict(X_test)

        return preds.T, X_test.index

    def _cv_strategy_check(self, df, return_compatible_strategies: bool = False):
        """
        Method that checks if the DataFrame is compatabile with the
        chosen cross-validation strategy.

        Args
        ------

        df: pd.DataFrame
            DataFrame to check. Should be the training set.

        cv_strategy: Union[int, PanelSplit]
            Cross-validation strategy to use.

        return_compatible_strategies: bool
            Whether to return the compatible cross-validation strategies.

        Returns
        ------

        if return_compatible_strategies is True:
            Union[None, List[Union[int, PanelSplit]]]
                List of compatible cross-validation strategies.

        else:
            ValueError if the DataFrame is not compatible with the chosen cross-validation strategy.
            Otherwise, prints a message to the user.
        """

        unique_df_time = sorted(df[self.time_col].unique())

        common_time = sorted(
            set(unique_df_time).intersection(set(self.unique_cal_time))
        )

        # check test_size is valid in the context of the window_size
        if len(common_time) == 0:
            raise ValueError(
                "Your dataframe is invalid. You need at least one time period "
                "from the calibration set. Please revise your choice of window_size."
            )

        elif len(common_time) <= 2:

            warnings.warn(
                "You are constrained to using standard cross-validation only. If you would like to use PanelSplit, please revise the arguments used to generate lags."
            )

            if return_compatible_strategies:
                return [int]

        else:
            print(
                "Your DataFrame is compatible with either standard cross validation or PanelSplit."
            )

            if return_compatible_strategies:
                return [int, PanelSplit]

    def _cal_time_check(self, df):
        """
        Method that checks if the DataFrame is valid for the chosen window_size.

        Args
        ------

        df: pd.DataFrame
            DataFrame to check. Should be the training set.

        Returns
        ------

        ValueError if the DataFrame is not valid for the chosen window_size.
        Otherwise, prints a message to the user.
        """

        unique_df_time = sorted(df[self.time_col].unique())

        common_time = sorted(
            set(unique_df_time).intersection(set(self.unique_cal_time))
        )

        # check test_size is valid in the context of the window_size
        if len(common_time) == 0:
            raise ValueError(
                "Your dataframe is invalid. You need at least one time period from the calibration set in order to fit the model and generate predictions for the first test set. Please revise the arguments used to generate lags and re-tune."
            )

        else:
            print(
                f"Everything is fine. The first test set prediction use a model fitted on {common_time}."
            )
    @staticmethod
    def process_split(args):
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
        return preds.T, X_test.index

    def nonconformity_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Method that calculates the nonconformity score of the model on a given dataset.
        Since this is a class for implementing LPCI on a regression model, the nonconformity score is the residuals.

        Parameters
        ----

        df: pd.DataFrame
            DataFrame containing the predictions of the model.
            It must contain the columns provided upon initialization: [unit_col, time_col, preds_col, true_col]
            The time_col should represent the time of the prediction, not the time of the true values.

        Returns
        ----

        pd.DataFrame
            DataFrame containing the nonconformity scores of the model.
        """

        df = df.copy()
        df["residuals"] = df[self.true_col] - df[self.preds_col]

        return df

    def lag(
        self,
        df: pd.DataFrame,
        col: str,
        lags: list,
        decay: float = None,
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

        decay: float
            Decay factor for exponential smoothing of the residuals. If None, no decay is applied.
        
        adjust: bool
            Whether to adjust the weights for the exponential moving average.
            Only used if decay is not None.
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

            #smooth the residuals if decay is not None
            if decay is not None:
                df[f'{col}_lag_{lag}'] = df.groupby(self.unit_col, observed = True)[f'{col}_lag_{lag}'].transform(lambda x: x.ewm(alpha = decay, adjust = adjust).mean())

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
        decay: float = None,
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

        decay: float
            Decay factor for exponential smoothing of the residuals. If None, no decay is applied.

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
            decay=decay,
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

        # warn the user of possible cross-validation strategies
        self._cv_strategy_check(df[df[self.time_col].isin(self.unique_cal_time)])

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
        grid_search_method: str = "GridSearchCV",
        grid_search_kwargs: dict = None,
        cv_kwargs: Union[int, dict] = None,
        return_best_estimator: bool = False,
    ):
        """
        Method that fits the quantile regression forest. GridSearchCV is used to tune the hyperparameters.
        We restrict the training data only to time periods before the last calibration time.
        We optionally allow for time series cross-validation using PanelSplit. This is preferred, but not strictly necessary.
        In cases where you have a small calibration dataset, eval_delay >=1 and/or prefer a larger window size, using panel_split may not be feasible.

        Args
        ------

        df: pd.DataFrame
            DataFrame containing the features and target variable.

        features: list
            List of features to include in the model.

        target_col: str
            Name of the target variable.

        alpha: float
            Significance level for the prediction interval.

        n_quantiles: int
            Number of quantiles to generate for each side of the prediction interval.

        grid_search_method: str
            Method to use for hyperparameter tuning. Currently, only 'GridSearchCV' and 'RandomizedSearchCV' is supported.

        grid_search_kwargs: dict
            Dictionary containing keyword arguments required to conduct a hyperparameter search.
            It should contain keyword arguments for GridSearchCV or RandomizedSearchCV. It must contain either:
            - 'param_grid': dict. Dictionary containing the hyperparameters to tune for GridSearchCV.
            - 'param_distributions': dict. Dictionary containing the hyperparameters to tune for RandomizedSearchCV.
            Remaining arguments are dicated by scikit-learn's GridSearchCV and RandomizedSearchCV documentation. Examples include:
            - 'scoring': str. Scoring metric to use. None means sklearn's default scoring metric (r2 for regression) is used. If you prefer a different metric, you must specify it to handle multi-output regression (because we are predicting quantiles).
            - 'n_jobs': int. Number of jobs to run in parallel.
            - 'n_iter': int. Number of iterations for RandomizedSearchCV.

        cv_kwargs: Union[int, dict]
            Cross-validation strategy to use. If an integer is provided, standard cross-validation is used.
            Otherwise, a dictionary containing the arguments for PanelSplit should be provided. 
            Critical keys include:
            - 'gap': int. Gap between train and test sets in PanelSplit.
            - 'test_size': int. Number of unique time periods in each test set.
            - 'n_splits': int. Number of splits to use in PanelSplit.

        return_best_estimator: bool
            Whether to return the best estimator.

        Returns
        ------
        best_params_: dict
            Dictionary containing the best hyperparameters.

        quantiles: np.array
            Array containing the quantiles for the prediction.

        best_estimator_: RandomForestQuantileRegressor (optional)
            Fitted quantile regression forest.
        """

        # Check that the time_col is of type int
        self._dtype_check(df, self.time_col, np.dtype("int64"))

        # Use only data from the calibration set
        train_df = (
            df[df[self.time_col].isin(self.unique_cal_time)]
            .sort_values(by=self.id_vars)
            .reset_index(drop=True)
        )
        compatible_cv_strategies = self._cv_strategy_check(
            train_df, return_compatible_strategies=True
        )

        if isinstance(cv_kwargs, dict) and PanelSplit not in compatible_cv_strategies:
            raise ValueError(
                "Your DataFrame is not compatible with PanelSplit. "
                "Please revise your choice of window_size and/or cross-validation strategy."
            )
        
        # Split into X_train and y_train
        X_train = train_df[features]
        y_train = train_df[target_col]

        # Generate the quantiles
        quantiles = self.gen_quantiles(alpha, n_quantiles)

        # Initialize the quantile regressor
        qrf = RandomForestQuantileRegressor(q=quantiles)

        #first generate the required cross-validation strategy
        if isinstance(cv_kwargs, int):
            cv = cv_kwargs

        elif isinstance(cv_kwargs, dict):
            cv = PanelSplit(
                train_df[self.time_col], 
                **cv_kwargs
            )

        else:
            raise ValueError("cv_kwargs must be an integer or a dictionary.")

        if grid_search_method == "GridSearchCV":
            required_keys = {"param_grid"}
            
            if not required_keys.issubset(grid_search_kwargs.keys()):
                raise ValueError(
                    f"grid_search_kwargs must contain the following keys: {required_keys}."
                )
            
            grid_search = GridSearchCV(
                qrf, cv=cv, **grid_search_kwargs
            )

        elif grid_search_method == "RandomizedSearchCV":

            required_keys = {"param_distributions"}

            if not required_keys.issubset(grid_search_kwargs.keys()):
                raise ValueError(
                    f"grid_search_kwargs must contain the following keys: {required_keys}."
                )
            
            grid_search = RandomizedSearchCV(
                    qrf, cv=cv, **grid_search_kwargs
                )
        
        # Fit the model
        grid_search.fit(X_train, y_train)

        if return_best_estimator:
            return grid_search.best_params_, quantiles, grid_search.best_estimator_
        else:
            return grid_search.best_params_, quantiles

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
        df:pd.DataFrame,
        features:list,
        target_col:str,
        best_params:dict,
        alpha:float,
        n_quantiles:int = 5,
        panel_split_kwargs:dict = None,
        n_jobs:int = -1,
        return_fitted_estimators:bool = False
        ):
        
        """
        Method that fits the quantile regression forest and generates the predictions.

        Args
        ------

        df: pd.DataFrame
            DataFrame containing the features and target variable.
        
        features: list
            List of features to include in the model.

        target_col: str
            Name of the target variable.

        best_params: dict
            Dictionary containing the best hyperparameters.

        alpha: float
            Significance level for the prediction interval.
        
        n_quantiles: int
            Number of quantiles to generate for either side of the prediction interval.

        panel_split_kwargs: dict
            A dictionary containing the arguments for PanelSplit should be provided. 
            Critical keys include:
            - 'gap': int. Gap between train and test sets in PanelSplit.
            - 'test_size': int. Number of unique time periods in each test set.
            There is no need to specify 'n_splits' as this is automatically calculated.

        n_jobs: int
            Number of jobs to run in parallel.

        return_fitted_estimators: bool
            Whether to return the fitted estimators.

        Returns
        ------
        """

        #check that the time_col is of type int
        self._dtype_check(df, self.time_col, np.dtype('int64'))

        # Generate the quantiles
        quantiles = self.gen_quantiles(alpha, n_quantiles)

        # Check that the DataFrame contains at least one time period from the calibration set
        self._cal_time_check(df)

        #initialize our panelsplit object
        panel_split = PanelSplit(
            df[self.time_col], 
            n_splits= self._get_n_splits(df[self.time_col].unique(), min(self.unique_test_time)), 
            **panel_split_kwargs
            )

        # Create an empty DataFrame to store the predictions
        interval_df = panel_split.gen_test_labels(df[self.id_vars + [target_col]])

        #merge back on the original predictions and target from the test set
        interval_df = interval_df.reset_index() #to preserve the index
        interval_df = interval_df.merge(self.test_preds[self.id_vars + [self.preds_col, self.true_col]], on = self.id_vars, how = 'left')
        interval_df = interval_df.set_index('index')
        interval_df.index.name = None

        # Initialize the quantile regressor
        qrf = RandomForestQuantileRegressor(q=quantiles, **best_params)

        # Perform cross-validation predictions
        fitted_estimators = panel_split.cross_val_fit(
            estimator=qrf,
            X=df[features].copy(),
            y=df[target_col].copy(),
        )

        if n_jobs == 1:
            # Sequential processing
            test_preds = []
            for i, (_, test_indices) in tqdm(
                enumerate(panel_split.split()), total=len(list(panel_split.split()))
            ):
                preds, index = self._predict_split(
                    fitted_estimators[i], df.loc[test_indices, features]
                )
                test_preds.append((preds, index))

        else:
            # Use multiprocessing for parallel processing
            ## CHANGE TO ACCEPT OTHER NEGATIVE VALUES
            if n_jobs <= -1:
                num_processes = os.cpu_count()
            elif n_jobs > 0:
                num_processes = n_jobs  # Use the specified number of jobs
            else:
                raise ValueError(f"Invalid n_jobs value: {n_jobs}")

            # Use multiprocessing for parallel processing
            args = [
                (fitted_estimators[i], df.loc[test_indices, features].copy(), i)
                for i, (_, test_indices) in enumerate(panel_split.split())
            ]
            with mp.Pool(processes=num_processes) as pool:
                test_preds = pool.map(LPCI.process_split, args)

        # Unpack the test_preds into the interval_df
        for (preds, index) in test_preds:
            
            interval_df.loc[index, [f"q_{np.round(i, 5)}" for i in quantiles]] = preds
            lower_conf, upper_conf, opt_lower_q, opt_upper_q = self.gen_conf_interval(
                        preds, quantiles
                    )
            interval_df.loc[index, f"{target_col}_lower_conf"] = lower_conf
            interval_df.loc[index, f"{target_col}_upper_conf"] = upper_conf
            interval_df.loc[index, "opt_lower_q"] = opt_lower_q
            interval_df.loc[index, "opt_upper_q"] = opt_upper_q

        # Add confidence intervals
        interval_df["lower_conf"] = interval_df[self.preds_col] + interval_df[
            f"{target_col}_lower_conf"
        ]
        interval_df["upper_conf"] = interval_df[self.preds_col] + interval_df[
            f"{target_col}_upper_conf"
        ]

        if return_fitted_estimators:
            return interval_df, fitted_estimators
        
        else:
            return interval_df
    
    