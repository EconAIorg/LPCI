"""
This module implements the evaluation methods for the LPCI model.
"""

from lpci import LPCI

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines as mlines
import seaborn as sns
import pandas as pd


class EvaluateLPCI:
    """
    This class is used to evaluate the performance of the LPCI model.
    It contains methods to compute coverage on different subsets of the data.
    It also contains methods to plot prediction intervals.

    Args
    ----
    lpci: LPCI
        An instance of the LPCI class

    alpha: float
        The significance level of the prediction intervals.

    interval_df: pd.DataFrame
        A DataFrame containing the lower and upper bounds of the prediction intervals.
        It should minimimally contain the columns:
        - unit_col: The column containing the unit of observation
        - time_col: The column containing the time of observation
        - preds_col: The column containing the predicted values
        - true_col: The column containing the true values
        - 'lower_conf': The column containing the lower bound of the prediction interval
        - 'upper_conf': The column containing the upper bound of the prediction interval

    """

    def __init__(self, lpci: LPCI, alpha: float, interval_df: pd.DataFrame):

        self.unit_col = lpci.unit_col
        self.time_col = lpci.time_col
        self.preds_col = lpci.preds_col
        self.true_col = lpci.true_col

        self.alpha = alpha
        self.conf_int = int((1 - alpha) * 100)

        # Merge the interval_df with the test_preds
        self.df = interval_df

    def _compute_coverage(self, df: pd.DataFrame):
        """
        Function to compute the coverage of the prediction intervals.
        Defined as the proportion of observations where the true value falls within the prediction interval.

        Args
        ----
        df: pd.DataFrame
            A DataFrame containing the lower and upper bounds of the prediction intervals.
            It should minimimally contain the columns:
            - true_col: The column containing the true values
            - 'lower_conf': The column containing the lower bound of the prediction interval
            - 'upper_conf': The column containing the upper bound of the prediction interval

        Returns
        -------
        coverage: float
            The coverage of the prediction intervals.
        """

        # Compute the coverage
        coverage = (df[self.true_col] >= df["lower_conf"]) & (
            df[self.true_col] <= df["upper_conf"]
        )
        coverage = coverage.mean()

        return coverage

    def _convert_ln_to_abs(self, df: pd.DataFrame, col: str):
        """
        Function that converts a variable in ln terms to absolute terms

        Args
        -----
        df: pd.DataFrame
            The DataFrame containing the variable to convert.

        col: str
            The column to convert to absolute terms.

        Returns
        ------
        pd.Series
            The variable in absolute terms.
        """

        return np.exp(df[col]) - 1

    def overall_coverage(self):
        """
        Compute the overall coverage of the prediction intervals.
        """

        return self._compute_coverage(self.df)

    def coverage_by_unit(self):
        """
        Compute the coverage of the prediction intervals by unit.
        """

        coverage = self.df.groupby(self.unit_col, observed=True).apply(
            self._compute_coverage
        )
        return coverage

    def coverage_by_time(self):
        """
        Compute the coverage of the prediction intervals by time.
        """

        coverage = self.df.groupby(self.time_col, observed=True).apply(
            self._compute_coverage
        )
        return coverage

    def coverage_by_bin(self, bins: list, bin_labels: str, ln_to_absolute: bool):
        """
        Function that compute coverage of the prediction intervals by bin.

        Args
        -----
        bins
            A list of bin edges to use for binning the true value.

        bin_labels
            A list of labels to use for the bins.

        ln_to_absolute
            A boolean indicating whether to convert the true values to absolute values before binning.

        Returns
        -------

        df:pd.DataFrame
            A DataFrame containing the coverage of the prediction intervals by bin.
        """

        df = self.df.copy()

        # convert to absolute terms if specified
        if ln_to_absolute:
            for col in [self.true_col, "lower_conf", "upper_conf"]:
                df[col] = self._convert_ln_to_abs(df, col)

        # Bin the true values
        df[f"{self.true_col}_bin"] = pd.cut(
            df[self.true_col], bins=bins, labels=bin_labels, right=False
        )

        coverage = df.groupby(f"{self.true_col}_bin", observed=True).apply(
            self._compute_coverage
        )

        return coverage

    def _plot_intervals(self, plot_df: pd.DataFrame, x_col: str):
        """
        Function that plots the prediction intervals.

        Args
        ----
        plot_df: pd.DataFrame
            A DataFrame containing the data to plot.
            It should minimimally contain the columns:
            - x_col: The column containing the x-axis values
            - true_col: The column containing the true values
            - 'lower_conf': The column containing the lower bound of the prediction interval
            - 'upper_conf': The column containing the upper bound of the prediction interval

        x_col: str
            The column to use for the x-axis

        Returns
        -------
        fig: matplotlib.figure.Figure
            The figure object containing the plot.
        """

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the true values
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=self.true_col,
            ax=ax,
            color="blue",
            marker="o",
            label="True",
            s=100,
        )

        # Plot the prediction intervals
        for x_var in plot_df[x_col].unique():
            x_var_df = plot_df[plot_df[x_col] == x_var]
            ax.fill_between(
                x_var_df[x_col],
                x_var_df["lower_conf"],
                x_var_df["upper_conf"],
                color="gray",
                alpha=0.25,
                linewidth=25,
            )

        # Rotate x-axis labels 90 degrees for readability
        # ax.set_xticklabels(plot_df[x_col], rotation=90)

        # Add labels and title
        ax.set_xlabel(x_col)
        ax.set_ylabel("Value")

        # add confidence interval legend
        line = mlines.Line2D(
            [],
            [],
            color="gray",
            alpha=0.25,
            label=f"{self.conf_int}% prediction interval",
            linewidth=5,
        )
        handles, labels = ax.get_legend_handles_labels()
        handles.append(line)
        labels.append(f"{self.conf_int}% prediction interval")

        ax.legend(handles, labels, fontsize=14, loc="upper left")

        # Adjust layout to make room for rotated labels
        plt.tight_layout()

        return fig

    def plot_intervals_year(self, year: int):
        """
        Plot the prediction intervals for a given year.

        Args
        ----
        year: int
            The year for which to plot the prediction intervals.
        """

        plot_df = self.df[self.df[self.time_col] == year].sort_values(by=self.unit_col)
        return self._plot_intervals(plot_df, self.unit_col)

    def plot_intervals_unit(self, unit: str):
        """
        Plot the prediction intervals for a given unit.

        Args
        ----
        unit: str
            The unit for which to plot the prediction intervals.
        """

        plot_df = self.df[self.df[self.unit_col] == unit].sort_values(by=self.time_col)
        return self._plot_intervals(plot_df, self.time_col)