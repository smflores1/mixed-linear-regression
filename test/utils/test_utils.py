"""

    Tests:
    ======
    Name: `test_utils.py`
    Author: Steven Flores
    Date: 2024-02-14

    Note:
    =====
    This is not finished. More tests have yet to be written.

"""

# External packages:
import pytest
import numpy as np
import scipy.stats as stats

# Local modules:
import utils.utils as utils

###################################################################################################
## Test setup:
###################################################################################################


class TestData:

    def __init__(self):

        np.random.seed(1)

        # Number of regressor and response variables:
        self.n_regressors = 3
        self.n_responses = 2

        # Number of samples:
        self.n_samples = 1000

        # Number of components (i.e., linear models):
        self.n_components = 2

        # Regressor parameters:
        self.mean1_vec = np.array([1, 2, 3])
        self.mean2_vec = np.array([4, 5, 6])

        assert len(self.mean1_vec) == self.n_regressors
        assert len(self.mean2_vec) == self.n_regressors

        self.reg_cov1_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.reg_cov2_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        assert self.reg_cov1_mat.shape == (self.n_regressors, self.n_regressors)
        assert self.reg_cov2_mat.shape == (self.n_regressors, self.n_regressors)

        # Linear model parameters:
        self.slope1_mat = np.array([[1, 2], [3, 4], [5, 6]])
        self.slope2_mat = np.array([[7, 8], [9, 10], [11, 12]])

        assert self.slope1_mat.shape == (self.n_regressors, self.n_responses)
        assert self.slope2_mat.shape == (self.n_regressors, self.n_responses)

        self.bias1_vec = np.array([1, 2])
        self.bias2_vec = np.array([3, 4]) + 1e3

        assert len(self.bias1_vec) == self.n_responses
        assert len(self.bias2_vec) == self.n_responses

        self.res_cov1_mat = 1e-2 * np.array([[1, 0], [0, 1]])
        self.res_cov2_mat = 1e-2 * np.array([[2, 0], [0, 3]])

        assert self.res_cov1_mat.shape == (self.n_responses, self.n_responses)
        assert self.res_cov2_mat.shape == (self.n_responses, self.n_responses)

        self.X1_mat = np.random.multivariate_normal(
            self.mean1_vec, self.reg_cov1_mat, self.n_samples
        )
        self.X2_mat = np.random.multivariate_normal(
            self.mean2_vec, self.reg_cov2_mat, self.n_samples
        )

        # Responses without errors:
        self.Y1_mat = np.dot(self.X1_mat, self.slope1_mat) + self.bias1_vec
        self.Y2_mat = np.dot(self.X2_mat, self.slope2_mat) + self.bias2_vec

        # Responses with errors:
        self.Y1_mat += np.random.multivariate_normal(
            np.zeros(len(self.res_cov1_mat)), self.res_cov1_mat, self.n_samples
        )
        self.Y2_mat += np.random.multivariate_normal(
            np.zeros(len(self.res_cov1_mat)), self.res_cov2_mat, self.n_samples
        )

        # Component responsabilities:
        self.resp1_mat = np.stack(
            [np.ones(self.n_samples), np.zeros(self.n_samples)], axis=1
        )
        self.resp2_mat = np.stack(
            [np.zeros(self.n_samples), np.ones(self.n_samples)], axis=1
        )

        # Concatenate the two components:
        self.X_mat = np.concatenate([self.X1_mat, self.X2_mat], axis=0)
        self.Y_mat = np.concatenate([self.Y1_mat, self.Y2_mat], axis=0)
        self.resp_mat = np.concatenate([self.resp1_mat, self.resp2_mat], axis=0)


@pytest.fixture
def test_data():

    return TestData()


###################################################################################################
## Tests:
###################################################################################################


def test_estimate_gaussian_parameters(test_data):

    gaussian_parameters = utils.estimate_gaussian_parameters(
        test_data.X_mat, test_data.resp_mat
    )

    # Tests:

    assert np.allclose(gaussian_parameters.mean_mat[0], test_data.mean1_vec, atol=1e-1)
    assert np.allclose(gaussian_parameters.mean_mat[1], test_data.mean2_vec, atol=1e-1)

    assert np.allclose(
        gaussian_parameters.covariance_tensor[0], test_data.reg_cov1_mat, atol=1e-1
    )
    assert np.allclose(
        gaussian_parameters.covariance_tensor[1], test_data.reg_cov2_mat, atol=1e-1
    )


def test_estimate_linear_parameters(test_data):

    linear_parameters = utils.estimate_linear_parameters(
        test_data.X_mat, test_data.Y_mat, test_data.resp_mat
    )

    # Tests:

    assert np.allclose(linear_parameters.bias_mat[0], test_data.bias1_vec, atol=1e-1)
    assert np.allclose(linear_parameters.bias_mat[1], test_data.bias2_vec, atol=1e-1)

    assert np.allclose(
        linear_parameters.slope_tensor[0], test_data.slope1_mat, atol=1e-1
    )
    assert np.allclose(
        linear_parameters.slope_tensor[1], test_data.slope2_mat, atol=1e-1
    )

    assert np.allclose(
        linear_parameters.covariance_tensor[0], test_data.res_cov1_mat, atol=1e-1
    )
    assert np.allclose(
        linear_parameters.covariance_tensor[1], test_data.res_cov2_mat, atol=1e-1
    )


def test_compute_log_gaussian_prob(test_data):

    for X_mat, mean_vec, covariance_mat in zip(
        [test_data.X1_mat, test_data.X2_mat],
        [test_data.mean1_vec, test_data.mean2_vec],
        [test_data.reg_cov1_mat, test_data.reg_cov2_mat],
    ):

        # Tests:

        log_prob_gaussian_test_vec = utils.compute_log_gaussian_prob(
            X_mat, mean_vec, covariance_mat
        )
        log_prob_gaussian_true_vec = np.log(
            stats.multivariate_normal(
                mean=mean_vec,
                cov=covariance_mat,
            ).pdf(X_mat)
        )

        assert np.allclose(log_prob_gaussian_test_vec, log_prob_gaussian_true_vec)
