
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

array_1D = typing.Optional[npt.NDArray[npt.Shape['*'], npt.Float]]
array_2D = typing.Optional[npt.NDArray[npt.Shape['*, *'], npt.Float]]
array_3D = typing.Optional[npt.NDArray[npt.Shape['*, *, *'], npt.Float]]

class CovarianceTensor:

    def __init__(
        self,
        covariance_tensor: array_3D,
    ):

        self._covariance_tensor = None
        self._precision_cholesky_tensor = None
        self._precision_tensor = None

        # This calls the setter:
        self.covariance_tensor = covariance_tensor

    @property
    def covariance_tensor(self) -> npt.NDArray[npt.Shape['*, *, *'], npt.Float]:
        return self._covariance_tensor

    @covariance_tensor.setter
    def covariance_tensor(
        self,
        tensor: array_3D
    ) -> None:
        self._covariance_tensor = tensor
        if tensor is None:
            self._precision_cholesky_tensor = None
            self._precision_tensor = None
        else:
            self._precision_cholesky_tensor = compute_precision_cholesky(tensor, 'full')
            self._precision_tensor = np.dot(
                self._precision_cholesky_tensor,
                self._precision_cholesky_tensor.T,
            )

    @property
    def precision_cholesky_tensor(self) -> npt.NDArray[npt.Shape['*, *, *'], npt.Float]:
        return self._precision_cholesky_tensor

    @property
    def precision_tensor(self) -> npt.NDArray[npt.Shape['*, *, *'], npt.Float]:
        return self._precision_tensor

class GaussianParameters(CovarianceTensor):

    def __init__(
        self,
        mean_mat: array_2D = None,
        covariance_tensor: array_3D = None,
    ) -> None:

        self.mean_mat = mean_mat
        super().__init__(covariance_tensor)

        self.n_components = None
        self.n_features = None

        if self.mean_mat is not None:
            self.n_components = self.mean_mat.shape[0]
            self.n_features = self.mean_mat.shape[1]
            self.mean_mat = check_means(
                self.mean_mat, self.n_components, self.n_features
            )

        # NOTE: it is misleading to use `check_precisions` on a covariance matrix,
        # but it works because precision and covariance matrices share the same properties:
        if self.covariance_tensor is not None:
            if self.n_components is None:
                self.n_components = self.covariance_tensor.shape[0]
            if self.n_features is None:
                self.n_features = self.covariance_tensor.shape[1]
            self.covariance_tensor = check_precisions(
                self.covariance_tensor,
                'full',
                self.n_components,
                self.n_features,
            )

class LinearParameters(CovarianceTensor):

    def __init__(
        self,
        bias_mat: array_2D = None,
        slope_tensor: array_3D = None,
        covariance_tensor: array_3D = None,
    ) -> None:

        self.bias_mat = bias_mat
        self.slope_tensor = slope_tensor
        super().__init__(covariance_tensor)

        self.n_components = None
        self.n_features = None
        self.n_responses = None

        if self.bias_mat is not None:
            self.n_components = self.bias_mat.shape[0]
            self.n_responses = self.bias_mat.shape[1]
            self.bias_mat = check_biases(
                self.bias_mat, self.n_components, self.n_responses
            )

        if self.slope_tensor is not None:
            if self.n_components is None:
                self.n_components = self.slope_tensor.shape[0]
            self.n_features = self.slope_tensor.shape[1]
            self.slope_tensor = check_slopes(
                self.slope_tensor, self.n_components, self.n_features, self.n_responses
            )

        # NOTE: it is misleading to use `check_precisions` on a covariance matrix,
        # but it works because precision and covariance matrices share the same properties:
        if self.covariance_tensor is not None:
            if self.n_components is None:
                self.n_components = self.covariance_tensor.shape[0]
            if self.n_responses is None:
                self.n_responses = self.covariance_tensor.shape[1]
            self.covariance_tensor = check_precisions(
                self.covariance_tensor,
                'full',
                self.n_components,
                self.n_responses,
            )

@dc.dataclass
class LinearModel:
    weight_vec: array_1D
    linear_parameters: typing.Optional[LinearParameters]
    gaussian_parameters: typing.Optional[GaussianParameters]

    def __post_init__(self):

        self.n_components = self.gaussian_parameters.n_components
        self.n_features = self.gaussian_parameters.n_features
        self.n_responses = self.linear_parameters.n_responses

        if self.weight_vec is not None:
            self.weight_vec = check_weights(self.weight_vec, self.n_components)

        # Check that the number of components match:
        if self.linear_parameters.n_components != self.n_components:
            raise ValueError(
                'Different number of components found in the regressor and response variables.'
            )

        # Check that the number of features match:
        if self.linear_parameters.n_features != self.n_features:
            raise ValueError(
                'Different number of features found in the regressor and response variables.'
            )

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

def check_shape(*args):
    return base._check_shape(*args)

def check_weights(*args):
    return gaussian_mixture._check_weights(*args)

def check_means(*args):
    return gaussian_mixture._check_means(*args)

def check_biases(
    bias_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    n_components: int,
    n_responses: int,
) -> npt.NDArray[npt.Shape['*, *'], npt.Float]:

    bias_mat = check_array(
        bias_mat,
        dtype=[np.float64, np.float32],
        ensure_2d=True,
    )
    check_shape(bias_mat, (n_components, n_responses), 'biases')

    return bias_mat

def check_slopes(
    slope_tensor: npt.NDArray[npt.Shape['*, *, *'], npt.Float],
    n_components: int,
    n_features: int,
    n_responses: int,
) -> npt.NDArray[npt.Shape['*, *, *'], npt.Float]:

    slope_tensor = check_array(
        slope_tensor,
        dtype=[np.float64, np.float32, np.float32],
        ensure_2d=False,
        allow_nd=True,
    )
    check_shape(slope_tensor, (n_components, n_features, n_responses), 'slopes')

    return slope_tensor

def check_precisions(*args):
    return gaussian_mixture._check_precisions(*args)

def compute_precision_cholesky(*args):
    return gaussian_mixture._compute_precision_cholesky(*args)

def estimate_gaussian_parameters(
    X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
) -> GaussianParameters:

    n_samples = X_mat.shape[0]
    n_features = X_mat.shape[1]
    n_components = resp_mat.shape[1]

    if resp_mat.shape[0] != n_samples:
        raise ValueError(
            'Different number of rows found in the regressor and responsibility matrices.'
        )

    resp_total_vec = resp_mat.sum(axis=0) + 10 * np.finfo(resp_mat.dtype).eps

    mean_mat = np.dot(resp_mat.T, X_mat) / resp_total_vec[:, np.newaxis]
    covariance_tensor = np.empty((n_components, n_features, n_features))

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

    if Y_mat.shape[0] != n_samples:
        raise ValueError(
            'Different number of rows found in the regressor and response matrices.'
        )

    if resp_mat.shape[0] != n_samples:
        raise ValueError(
            'Different number of rows found in the regressor and responsibility matrices.'
        )

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

def fit_linear_model(
    X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
) -> LinearParameters:

    n_samples = X_mat.shape[0]

    if Y_mat.shape[0] != n_samples:
        raise ValueError(
            'Different number of rows found in the regressor and response matrices.'
        )

    if resp_mat.shape[0] != n_samples:
        raise ValueError(
            'Different number of rows found in the regressor and responsibility matrices.'
        )

    weight_vec = np.mean(resp_mat, axis=0)
    linear_parameters = estimate_linear_parameters(X_mat, Y_mat, resp_mat)
    gaussian_parameters = estimate_gaussian_parameters(X_mat, resp_mat)

    return LinearModel(
        weight_vec=weight_vec,
        linear_parameters=linear_parameters,
        gaussian_parameters=gaussian_parameters,
    )

def compute_log_gaussian_prob(
    X_mat: array_2D,
    mean_vec: array_1D,
    covariance_mat: array_2D,
):

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
