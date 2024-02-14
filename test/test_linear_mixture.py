


import pytest
import numpy as np
import scipy.stats as stats
import scipy.special as special

import linear_mixture as lm
import utils.utils as utils
import test.utils.test_utils as test_utils


@pytest.fixture
def test_data():

    return test_utils.TestData()


@pytest.fixture
def linear_mixture():

    return lm.LinearMixture(
        n_components=2,
        tol=1e-3,
        max_iter=100,
    )

def test__initialize_parameters(test_data, linear_mixture):

    # Clusters are sufficiently far apart that k-means distinguishes them completely.

    random_state = 1

    linear_model = utils.fit_linear_model(test_data.X_mat, test_data.Y_mat, test_data.resp_mat)

    # A first attempt to find clusters using k-means and fit a linear model to each:
    linear_mixture._initialize_parameters(test_data.X_mat, test_data.Y_mat, random_state)

    # Check that Gaussian model for regressors agree:
    assert np.allclose(linear_mixture.means_, linear_model.gaussian_parameters.mean_mat)
    assert np.allclose(linear_mixture.regressor_covariances_, linear_model.gaussian_parameters.covariance_tensor)

    # Check that linear model for responses agree:
    assert np.allclose(linear_mixture.biases_, linear_model.linear_parameters.bias_mat)
    assert np.allclose(linear_mixture.slopes_, linear_model.linear_parameters.slope_tensor)
    assert np.allclose(linear_mixture.response_covariances_, linear_model.linear_parameters.covariance_tensor)

    # Check the weights. With the clusters separated and having the same number
    # of points, the probability of belonging to either cluster should be 0.5:
    assert np.allclose(linear_mixture.weights_, np.array([0.5, 0.5]))

def test__estimate_log_prob(test_data, linear_mixture):

    random_state = 1

    # Concatenate the two components:
    X_mat = np.concatenate([test_data.X1_mat, test_data.X2_mat], axis=0)
    Y_mat = np.concatenate([test_data.Y1_mat, test_data.Y2_mat], axis=0)

    resp_mat = np.concatenate([test_data.resp1_mat, test_data.resp2_mat], axis=0)

    linear_model = utils.fit_linear_model(X_mat, Y_mat, resp_mat)

    # A first attempt to find clusters using k-means and fit a linear model to each:
    # TODO: is there a way to not have to explicitly call this each time?
    linear_mixture._initialize_parameters(X_mat, Y_mat, random_state)

    prob_xy_given_component_mat = linear_mixture._estimate_log_prob(
        X_mat, Y_mat,
    )
    prob_xy_given_component_mat = np.exp(prob_xy_given_component_mat)

    # Iterate over observations:
    for prob_xy_given_component_vec, X_vec, Y_vec in zip(
        prob_xy_given_component_mat, X_mat, Y_mat,
    ):
        # Iterate over components:
        for component_index, prob_xy_given_component_test in enumerate(
            prob_xy_given_component_vec
        ):

            # Probability of regressors given the component:
            prob_density_x_given_component = stats.multivariate_normal(
                mean=linear_model.gaussian_parameters.mean_mat[component_index],
                cov=linear_model.gaussian_parameters.covariance_tensor[component_index],
            ).pdf(X_vec)

            # Probability of responses given regressors and the component:
            res_vec = (
                Y_vec - np.dot(X_vec, linear_model.linear_parameters.slope_tensor[component_index])
                - linear_model.linear_parameters.bias_mat[component_index]
            )
            prob_density_y_given_x_and_component = stats.multivariate_normal(
                    mean=np.zeros(test_data.n_responses),
                cov=linear_model.linear_parameters.covariance_tensor[component_index],
            ).pdf(res_vec)

            # Probability of responses and regressors given the component:
            prob_xy_given_component_true = (
                prob_density_y_given_x_and_component * prob_density_x_given_component
            )

            assert abs(prob_xy_given_component_test - prob_xy_given_component_true) < 1e-10

def test__estimate_log_prob_resp(test_data, linear_mixture):

    random_state = 1

    # A first attempt to find clusters using k-means and fit a linear model to each:
    linear_mixture._initialize_parameters(test_data.X_mat, test_data.Y_mat, random_state)

    # Responsibity matrix (`self.n_samples` x `self.n_components`):
    sum_comp_log_prob_xy_vec, log_resp_mat = linear_mixture._estimate_log_prob_resp(
        test_data.X_mat, test_data.Y_mat,
    )

    # Recompute responsibilities (i.e., probability of component given sample):
    weighted_log_prob_xy_mat = linear_mixture._estimate_weighted_log_prob(
        test_data.X_mat, test_data.Y_mat,
    )
    for weighted_log_prob_xy_vec, log_resp_vec, sum_comp_log_prob_xy in zip(
        weighted_log_prob_xy_mat, log_resp_mat, sum_comp_log_prob_xy_vec,
    ):

        assert sum_comp_log_prob_xy == special.logsumexp(weighted_log_prob_xy_vec)

        for weighted_log_prob_xy, log_resp in zip(weighted_log_prob_xy_vec, log_resp_vec):

            assert log_resp == weighted_log_prob_xy - special.logsumexp(weighted_log_prob_xy_vec)

# TODO: unit tests for _estimate_log_weights and _estimate_weighted_log_prob?