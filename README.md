

# LPCI algorithm: a Conformal Inference method for panel data regression

This algorithm, based on the confromal inference framework, is used to obtain prediction intervals (quantify uncertainty) when working with panel data in a regression setting. 

## Installation 

You can install the package using pip: 

```
pip install lpci
``` 


## Introduction

The package implements the Longitudinal Prediction Conformal Inference (LPCI) algorithm presented by Devesh Batra, Salvatore Mercuri & Raad Khraishi in the paper "Conformal Predictions for Longitudinal Data" (https://arxiv.org/abs/2310.02863).

The authors prove that the LPCI method asymptotically ensures both longitudinal conditional coverage and marginal cross-sectional coverage. In theory, with sufficient data points, both types of coverage should be at least equal to the confidence level.

Below we provide a mathematical overview, much of which draws directly from Batra et al. (2023). For those seeking a guide to practical implementation, please refer to 
[notebook](./notebooks/tutorial.ipynb).


## Model assumptions

We consider a dataset consisting of observations ${(X_t^{(g)}, Y_t^{(g)})\}_{t=1}^T$ of length $T > 1$ for each group g $\in G = \{1, \dots, |G|\}$, where:
- $$Y_t^{(g)} \in \mathbb{R}$$ is a continuous scalar representing the target variable for group $g$ at time $t$.
- $$X_t^{(g)} \in \mathbb{R}^d$$ consists of d-dimensional features associated with group $g$ at time $t$.

This package enables generating prediction intervals for panel data in a regression setting, where observations are indexed both by time $t$ and group $g$.

Data points are exchangeable if the joint probability distribution is invariant to any permutation of them. This assumption is usually what enables the conformal prediction framework to theoretically prove that coverage guarantees are ensured and are at least equal to the confidence level. Nonetheless, in a panel data setting where we have temporal dependence, the exchageability assumption does not hold. The LPCI is a framework where we can provide asymptotic coverage guarantees - cross-sectional (marginal) and longitudinal (conditional) - beyond the exchageability assumption. The authors make the reasonable assumption that the groups are exchangeable. 

The LPCI algorithm uses the split or inductive conformal inference method, that is, data should be seperated into three sets: training, calibration and test. The main idea is to use the non-conformity score (e.g. residuals) in the calibration set to obtain uncertainty intervals for the test points.

Further, in alignment with other conformal prediction methods, is model-agnostic i.e. it can be applied irrespective of the algorithm used to generate point predictions.

## LPCI algorithm 

The general procedure is as follows:

### 1: Train model and generate point predictions for the calibration set 

Split the dataset into three sets - training, calibration and testing:

$$
\mathcal{D}_{\text{train}} = \{ (X_t^{(g)}, Y_t^{(g)}) \mid t \in \text{train} \}
$$

$$
\mathcal{D}_{\text{cal}} = \{ (X_t^{(g)}, Y_t^{(g)}) \mid t \in \text{cal} \}
$$

$$
\mathcal{D}_{\text{test}} = \{ (X_t^{(g)}, Y_t^{(g)}) \mid t \in \text{test} \}
$$

Train a model $\widehat{f}$ on $\mathcal{D}_{\text{train}}$ and generate point predictions for the calibration set: 

$$
\widehat{Y}\_{t}^{(g)} = \widehat{f}(X\_{t}^{(g)}), \quad \text{for } X\_{t}^{(g)} \in \mathcal{D}_{\text{cal}}
$$


### 2: Compute non-conformity score 

Compute the non-confomity score for each observation in the calibration set; in our case, the residuals by default. The non-conformity score is a measure of how unusual or strange a prediction is according to the previous examples in the training data. 

$$
\widehat{\epsilon}\_t^{(g)} = Y\_t^{(g)} - \widehat{Y}\_t^{(g)}, \quad \text{for } Y\_t^{(g)} \in \mathcal{D}_{\text{cal}}
$$

The non-conformity score (residuals) for each observation in the calibration set will form the target variable for training a Quantile Random Forest and generating prediction intervals.

### 3: Input QRF (Features)

The `prepare_df` method enables three types of features to be generated:

#### Lagged Residuals

Lagged residuals are generated according to a specified window size w. Two options are available:

- *Simple Lags*: Directly using lags of past residuals over the specified window.

Using simple lags means that, for each group $𝑔$ at each time step $t$, we have:

$$
\mathcal{E}\_{t,w}^{(g)} := \left(\widehat{\epsilon}\_{t-1}^{(g)}, \ldots, \widehat{\epsilon}\_{t-w}^{(g)}\right) \in \mathbb{R}^w
$$

- *Exponential Smoothing*: Optionally, the package can compute exponentially-weighted mean residuals for each group.

If exponential smoothing is true, for each group $𝑔$ at each time step $t$, we compute the exponentially-weighted mean residuals over a fixed window size $w$:

$$
\mathcal{E}\_{t,w}^{(g)} := \left(\overline{\epsilon}\_{t-1}^{(g)}, \ldots, \overline{\epsilon}\_{t-w}^{(g)}\right) \in \mathbb{R}^w
$$

For details on how $$\overline{\epsilon}\_{k}^{(g)}$$ is computed, please refer to the pandas documentation on the pandas.DataFrame.ewm method at https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html.

#### Group Identifier

Unique group identifiers should also be included for each group. The only method currently supported is one-hot encoding. 

#### Exogenous Features

Additional exogenous features can be included if they are relevant to the modeling task.

### Step 4: Tune Quantile Random Forest

This step involves training a Quantile Regression Forest (QRF) model to estimate conditional quantiles of the residuals. The QRF is trained on the features generated in step 3. 

#### Quantile Regression Forest:

Typically, machine-learning algorithms produce only a point prediction (e.g. the mean). Conversely, QRF's generate a prediction of the conditional quantiles for a given input. Our use case implements a QRF to model the distribution of $\widehat{\epsilon}\_t^{(g)}$ conditional on $\mathcal{E}\_{t,w}^{(g)}$.

$$ 
\widehat{Q}\_{t,p}^{(g)} = QRF\_{p}(\widehat{\epsilon}\_t^{(g)} \mid \mathcal{E}_{t,w}^{(g)}) \quad \text{where } p \in (0, 1) 
$$

The main finding of Batra et al. (2023) is that the estimated quantiles $\widehat{Q}\_{t,p}^{(g)}$ from the QRF converge uniformly to the true quantiles $Q_{t,p}^{(g)}$ as the amount of training data increases, where the true quantile is defined as:

$$
Q_{t,p}^{(g)} := \inf \lbrace e \in \mathbb{R} \mid F(e \mid \mathcal{E}_{t,w}^{(g)}) \geq p \rbrace \quad \text{where } p \in (0, 1) 
$$

#### Hyperparameter Tuning:

To optimize the QRF model, hyperparameters are tuned using standard Cross-Validation or the PanelSplit package (https://github.com/4Freye/panelsplit) which ensures robust tuning while avoiding information leakage.


### 5: Construct prediction intervals 

Once the QRF is trained, prediction intervals are constructed for the test set. Recall that we already have point predictions for all observations in the test set:

$$
\widetilde{Y}\_{t}^{(g)} = \widehat{f}(X\_{t}^{(g)}), \quad \text{for } X\_{t}^{(g)} \in \mathcal{D}_{\text{test}}
$$

We then obtain quantile estimates of the non-conformity score using the trained QRF for each test point as: 

$$
\widetilde{Q}\_{t,p}^{(g)} = QRF\_{p}(\mathcal{E}_{t,w}^{(g)}) \quad \text{where } p \in (0, 1) 
$$

The intervals combine point predictions from the base model with these quantile estimates.

#### 5.1 Interval definition: 

For a test point t, the prediction interval is: 

$$
\widetilde{C}\_{t\-1}(X\_t^{(g)}) := \[\widetilde{Y}\_t^{(g)} + \widetilde{Q}\_{t,\beta}^{(g)}, \widetilde{Y}\_t^{(g)} + \widetilde{Q}\_{t,1\-\alpha+\beta}^{(g)}\]
$$

- $\widetilde{Q}\_{t,p}^{(g)}$ and $\widetilde{Q}\_{t,1\-\alpha+\beta}^{(g)}$: lower and upper bound of the quantiles estimated by the QRF.  

#### 5.2 Optimize interval width: 

The $\beta \in [0,\alpha]$ is adjusted to select the narrowest prediction interval (pair of quantiles) that guarantees coverage: 

$$
\beta = \arg\min\_{p \in [0,\alpha]} \lbrace \widehat{Q}\_{t,1\-\alpha+p}^{(g)} - \widehat{Q}\_{t,p}^{(g)} \rbrace
$$

### 6: Evaluation

#### 6.1 Coverage intution
In conformal inference, coverage measures how often the true outcome Y falls within the predicted intervals. 
Besides overall coverage, in panel (longitudinal) data, there are another two main types of coverage useful to assess uncertainty quantification performance: 

- *Cross-sectional (marginal) coverage*: Measures coverage across groups g for a fixed time point t. Intuitively, the fraction of different groups whose outcomes lie within the prediction intervals for each time stamp should be at least $(1-\alpha)$. 

- *Longitudinal (conditional) coverage*: Focuses on how well the intervals capture outcomes over time for each individual group. Specifically, within each group 
g, the fraction of times the actual outcome lies within the interval (conditional on the features 𝑋) should be at least $(1-\alpha)$.

#### 6.2 Coverage definitions

##### Definition 1 (asymptotic cross sectional coverage):


The conformal intervals $\hat{C}\_{t-1}(X^{(g)}\_{t})$ have asymptotic cross-sectional coverage if, for all $\epsilon > 0$, there exists $T_{0}$ such that:

$$
\Pr(Y_t^{(g)} \in \widehat{C}_{t-1}(X_t^{(g)})) > 1 - \alpha - \varepsilon,
$$

for all $t > T_{0}$ and $g \in G$. 

Cross-sectional coverage is marginal over the groups for a fixed given (large enough) time-point.


##### Defintion 2 (asymptotic longitudinal coverage):

We say that the conformal intervals $\widehat{C}_{t-1}(X_t^{(g)})$ have asymptotic longitudinal coverage *for a group g*, if:

$$
\Pr(Y_t^{(g)} \in \widehat{C}_{t-1}(X_t^{(g)}) \mid X_t^{(g)}) \to 1 - \alpha \text{ uniformly as } T \to \infty.
$$

Longitudinal coverage is asymptotic in t and conditional over the temporal dimension.

## Background

Work on LPCI algorithm started at [EconAI](https://www.linkedin.com/company/econ-ai/) in September 2024.

## Contributing

We welcome contributions from the community! Whether it’s reporting a bug, suggesting a feature, improving documentation, or submitting a pull request, your input is highly valued.


## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

