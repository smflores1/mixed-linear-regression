# TODO: type hinting everywhere
# TODO: docstrings everywhere
# TODO: specify dimensions of tensors when type hinting.
# TODO: change the name 'regressor' to 'feature'.
# TODO: type hinting

# Standard library:
import typing
import dataclasses as dc

# External modules:
import numpy as np
import nptyping as npt
import scipy.linalg as linalg
from sklearn.utils import check_array
import sklearn.mixture._base as base
import sklearn.mixture._gaussian_mixture as gaussian_mixture
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


def check_n_samples(
    X_mat: npt.NDArray[typing.Any, npt.Float],
    Y_mat: npt.NDArray[typing.Any, npt.Float],
    name_x: str,
    name_y: str,
):

    len_x = len(X_mat)
    len_y = len(Y_mat)
    if len_x != len_y:
        raise ValueError(
            f'The {name_x} and {name_y} matrices should have the same number '
            f'of samples (i.e. rows) but have {len_x} and {len_y} respectively.'
        )

def check_weights(*args):
    return gaussian_mixture._check_weights(*args)

def check_means(*args):
    return gaussian_mixture._check_means(*args)

def check_precisions(*args):
    return gaussian_mixture._check_precisions(*args)

def check_precisions_cholesky(
    precisions_cholesky_tensor: npt.NDArray[npt.Shape['*, *, * '], npt.Float],
    n_components: int,
    n_features: int,
):

    # Check the shape:
    base._check_shape(
        precisions_cholesky_tensor,
        (n_components, n_features, n_features),
        'Cholesky matrix of precision',
    )

    # Check that each Cholesky matrix ...:
    for component_index, cholesky_mat in enumerate(precisions_cholesky_tensor):
        # ...is lower triangular:
        if not np.allclose(cholesky_mat, np.tril(cholesky_mat)):
            raise ValueError(
                f'In component {component_index}, the Cholesky matrix of the '
                'precision matrix must be lower triangular, but it is not.'
            )
        # ...has only positive entries on the diagonal:
        if not np.all(np.diag(cholesky_mat) > 0):
            raise ValueError(
                f'In component {component_index}, the Cholesky matrix of the '
                'precision matrix can have only positive diagonal entries, but does not.'
            )



def check_shape(*args):
    return base._check_shape(*args)

def compute_precision_cholesky(*args):
    return gaussian_mixture._compute_precision_cholesky(*args)

def compute_precision_cholesky_from_precisions(*args):
    return gaussian_mixture._compute_precision_cholesky_from_precisions(*args)


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

# TODO: import the method _compute_precision_cholesky from gaussian_mixture and define it as a function here...

def compute_log_gaussian_prob(X_mat, mean_vec, covariance_mat):

    n_features = X_mat.shape[1]

    try:
        cov_chol = linalg.cholesky(covariance_mat, lower=True)
    except linalg.LinAlgError:
        raise ValueError()
    precisions_chol = linalg.solve_triangular(
        cov_chol, np.eye(len(covariance_mat)), lower=True
    ).T
    log_prob_vec = np.dot(X_mat, precisions_chol) - np.dot(mean_vec, precisions_chol)
    log_prob_vec = np.sum(np.square(log_prob_vec), axis=1)

    log_det = np.sum(np.log(np.diag(precisions_chol)))

    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob_vec) + log_det
