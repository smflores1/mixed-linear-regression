# TODO: type hinting everywhere
# TODO: docstrings everywhere
# TODO: specify dimensions of tensors when type hinting.
# TODO: change the name 'regressor' to 'feature'.
# TODO: type hinting

# Standard library:
import dataclasses as dc

# External modules:
import numpy as np
import nptyping as npt
from sklearn.utils import check_array
from sklearn.mixture._base import _check_shape
# from sklearn.mixture._gaussian_mixture import

@dc.dataclass
class GaussianParameters:
    mean_mat: npt.NDArray[npt.Shape['*, *'], npt.Float]
    covariance_tensor: npt.NDArray[npt.Shape['*, *, *'], npt.Float]

    def __post_init__(self):

        self.n_components = self.mean_mat.shape[0]
        self.n_features = self.mean_mat.shape[1]

        # Check that the number of components match:
        assert self.covariance_tensor.shape[0] == self.n_components

        # Check that the number of features match:
        assert self.covariance_tensor.shape[1] == self.n_features
        assert self.covariance_tensor.shape[2] == self.n_features

@dc.dataclass
class LinearParameters:
    bias_mat: npt.NDArray[npt.Shape['*, *'], npt.Float]
    slope_tensor: npt.NDArray[npt.Shape['*, *, *'], npt.Float]
    covariance_tensor: npt.NDArray[npt.Shape['*, *, *'], npt.Float]

    def __post_init__(self):

        self.n_components = self.bias_mat.shape[0]
        self.n_responses = self.bias_mat.shape[1]

        # Check that the number of components match:
        assert self.slope_tensor.shape[0] == self.n_components
        assert self.covariance_tensor.shape[0] == self.n_components

        # Check that the number of responses match:
        assert self.slope_tensor.shape[2] == self.n_responses
        assert self.covariance_tensor.shape[1] == self.n_responses
        assert self.covariance_tensor.shape[2] == self.n_responses

@dc.dataclass
class LinearModel:
    weight_vec: npt.NDArray[npt.Shape['*'], npt.Float]
    linear_parameters: LinearParameters
    gaussian_parameters: GaussianParameters

    def __post_init__(self):

        self.n_components = self.gaussian_parameters.n_components
        self.n_features = self.gaussian_parameters.n_features
        self.n_responses = self.linear_parameters.n_responses

        # Check that the number of components match:
        assert self.linear_parameters.n_components == self.n_components

        # Check that the number of features match:
        assert self.linear_parameters.slope_tensor.shape[1] == self.n_features

# TODO: DELETE
# def _check_weights(
#     weight_vec,
#     n_components,
# ):

#     # TODO: give credit to original authors.

#     weight_vec = check_array(weight_vec, dtype=[np.float64, np.float32], ensure_2d=False)
#     _check_shape(weight_vec, (n_components,), 'weights')

#     # Check range:
#     weight_min = np.min(weight_vec)
#     weight_max = np.max(weight_vec)
#     if weight_min < 0.0 or weight_max > 1.0:
#         raise ValueError(
#             f'The parameter 'weights' should be in the range '
#             '[0, 1], but got max value {weight_max}, min value {weight_min}.'
#         )

#     # Check normalization:
#     weight_sum = np.sum(weight_vec)
#     if weight_sum != 1.0:
#         raise ValueError(
#             'The parameter "weights" should be normalized, but got "sum(weights) = {weight_sum}".'
#         )

#     return weight_vec


def estimate_gaussian_parameters(
    X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
) -> GaussianParameters:

    n_samples = X_mat.shape[0]
    n_features = X_mat.shape[1]
    n_components = resp_mat.shape[1]

    assert resp_mat.shape[0] == n_samples

    resp_total_vec = resp_mat.sum(axis=0) + 10 * np.finfo(resp_mat.dtype).eps

    mean_mat = np.dot(resp_mat.T, X_mat) / resp_total_vec[:, np.newaxis]
    covariance_tensor = np.empty((n_components, n_features, n_features))

    # TODO: comput_resp_total within the loop by iterating over the rows of resp_total_vec:
    for component_index, (mean_vec, resp_total) in enumerate(zip(mean_mat, resp_total_vec)):
        diff_mat = X_mat - mean_vec
        covariance_tensor[component_index] = (
            np.dot(resp_mat[:, component_index] * diff_mat.T, diff_mat) / resp_total
        )

    return GaussianParameters(
        mean_mat=mean_mat,
        covariance_tensor=covariance_tensor,
    )


def estimate_linear_parameters(
    X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
) -> LinearParameters:

    n_samples = X_mat.shape[0]
    n_features = X_mat.shape[1]
    n_responses = Y_mat.shape[1]
    n_components = resp_mat.shape[1]

    assert Y_mat.shape[0] == n_samples
    assert resp_mat.shape[0] == n_samples

    X_mat = np.concatenate([X_mat, np.ones(n_samples).reshape(-1, 1)], axis=1)

    bias_mat = np.empty((n_components, n_responses))
    slope_tensor = np.empty((n_components, n_features, n_responses))
    covariance_tensor = np.empty((n_components, n_responses, n_responses))

    for component_index, resp_vec in enumerate(resp_mat.T):

        W_mat = np.diag(resp_vec)

        B_mat = np.linalg.inv(np.dot(X_mat.T, np.dot(W_mat, X_mat)))
        B_mat = np.dot(np.dot(np.dot(B_mat, X_mat.T), W_mat), Y_mat)

        bias_mat[component_index] = B_mat[-1]
        slope_tensor[component_index] = B_mat[:-1]
        residual_mat = Y_mat - np.dot(X_mat, B_mat)
        covariance_tensor[component_index] = np.dot(residual_mat.T, np.dot(W_mat, residual_mat))
        covariance_tensor[component_index] /= np.sum(resp_vec)

    return LinearParameters(
        bias_mat=bias_mat,
        slope_tensor=slope_tensor,
        covariance_tensor=covariance_tensor,
    )

# TODO: this also needs to return weights:
def fit_linear_model(
    X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
) -> LinearParameters:

    n_samples = X_mat.shape[0]

    assert Y_mat.shape[0] == n_samples
    assert resp_mat.shape[0] == n_samples

    weight_vec = np.mean(resp_mat, axis=0)
    linear_parameters = estimate_linear_parameters(X_mat, Y_mat, resp_mat)
    gaussian_parameters = estimate_gaussian_parameters(X_mat, resp_mat)

    return LinearModel(
        weight_vec=weight_vec,
        linear_parameters=linear_parameters,
        gaussian_parameters=gaussian_parameters,
    )

def compute_log_prob_gaussian(X_mat, mean_vec, covariance_mat):

    n_samples = X_mat.shape[0]
    n_features = X_mat.shape[1]

    assert len(mean_vec) == n_features
    assert covariance_mat.shape == (n_features, n_features)

    diff_mat = X_mat - mean_vec
    exponent_vec = np.empty(n_samples)
    for k, diff_vec in enumerate(diff_mat):
        exponent_vec[k] = -0.5 * np.dot(diff_vec, np.dot(np.linalg.inv(covariance_mat), diff_vec.T))
    log_prefactor = -0.5 * (n_features * np.log(2 * np.pi) + np.log(np.linalg.det(covariance_mat)))

    return log_prefactor + exponent_vec


# def estimate_component_weights(
#     resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float]
# ):
