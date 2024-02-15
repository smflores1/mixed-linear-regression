# Mixed Linear Regression

### Description
This repository provides an implementation of "mixed linear regression" (MLR) in python and with a scikit-learn interface. In mixed linear regression, multiple linear models, each called a "component", appear in training data. The objective is to identify the components and fit the data of each component to its own linear model.

An apparent way to accomplish the aforementioned goal is to proceed in two steps:
1. Use a traditional clustering algorithm to identify components.
2. Fit the data in each component to its own linear model.

Unfortunately, this approach does not always work, as the example below shows. Indeed, in mixed linear regression, the features in the training data split into regressors (i.e., independent variables) and responses (i.e., dependent variables), with the latter a linear function of the former plus a zero-mean Gaussian noise. However, traditional clustering algorithms assume no such relationship between features in the data. As a result, step 1. can fail to identify the components correctly.

To address this shortcoming, we do not separate steps 1 and 2 above. Instead, we use a Gaussian mixture model approach, but with the following adjustments:
1. We use Gaussian models for the regressor variables (with component-wise mean and covariance parameters).
2. We use linear models with zero-mean Gaussian noise for the response variables (component-wise bias, slope, and covariance parameters).

The result is what we call a "linear mixture model", and to fit such a model is what we call MLR. We estimate the model parameters with values that maximize the expectation of a log-likelihood function for the model.

### Example

The following example

### Usage
