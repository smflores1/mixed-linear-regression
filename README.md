# Mixed Linear Regression

### Description
This repository provides an implementation of "mixed linear regression" (MLR) in python and with a scikit-learn interface. In MLR, multiple linear models, each called a "component", appear in training data. The objective is to identify the components and fit the data of each component to its own linear model.

An apparent way to accomplish the aforementioned goal is to proceed in two steps:
1. Use a traditional clustering algorithm to identify components.
2. Fit the data in each component to its own linear model.

Unfortunately, this approach does not always work, as the example below shows. Indeed, in MLR, the features in the training data split into regressors (i.e., independent variables) and responses (i.e., dependent variables), with the latter a linear function of the former plus a zero-mean Gaussian noise. However, traditional clustering algorithms assume no such relationship between features in the data. As a result, step 1. can fail to identify the components correctly.

To address this shortcoming, we do not separate steps 1 and 2 above. Instead, we use a Gaussian mixture model approach, but with the following adjustments:
1. We use per-component Gaussian models for the regressor variables (with component-wise mean and covariance parameters).
2. We use per-component linear models with zero-mean Gaussian noise for the response variables (component-wise bias, slope, and covariance parameters).

The result is what we call a "linear mixture model", and to fit such a model to training data is what we call MLR. By "fit", we mean that we estimate the model parameters given above with values that maximize the weighted expectation of a certain log-likelihood function.

### Documentation
The interface is almost completely identical to the Gaussian mixture model implementation in [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html). See `linear_mixture.py` for more information.

### Example

The following example shows how a traditional Gaussian mixture model can fail to identify the two linear models but a linear mixture model can succeed. To reproduce it run
```
python example.py
```

### Usage
For training data of regressors `train_X_mat` and responses `train_Y_mat`, train the model as follows:
```
    lm = linear_mixture.LinearMixture(
        n_components=K
    ).fit(train_X_mat, train_Y_mat)
```
Fit parameters may be accessed thus:
```
lm.weights_  # Probability of component membership
lm.means_  # Per-component means of the Gaussian models for the regressor variables
lm.regressor_covariances_  # Per-component covariances of the Gaussian models for the regressor variables
lm.biases_  # Per-component biases of the linear model for the response variables
lm.slopes_  # Per-component slopes of the linear model for the response variables
lm.response_covariances_  # Per-component covariances of the Gaussian noise of the linear model for the response variables
```

To predict component labels on test data of regressors `test_X_mat` and responses `test_Y_mat`, do the following:
```
lm_label_vec = lm.predict(test_X_mat, test_Y_mat)
```