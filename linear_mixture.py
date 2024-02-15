'''

    Module:
    =======
    Name: `linear_mixture.py`
    Author: Steven Flores
    Date: 2024-02-14

    Description:
    ============
    This module is home to the `LinearMixture` class. An instance of this
    class fits training data to a linear mixture model, also called a 'mixed
    linear regression' or 'MLR' in the literature, via expectation maximization.
    That is, given training data of regressor variables `train_X_mat` and response
    variables `train_Y_mat`, `K` components (i.e., 'clusters'), each belonging to
    a distinct linear model, are found in the data by calling

    ```
        lm = linear_mixture.LinearMixture(
            n_components=K
        ).fit(train_X_mat, train_Y_mat)
    ```

    Based on this fit, component assignment predictions are made on test data
    of regressor variables `test_X_mat` and response variables `test_Y_mat` via

    ```
    lm_label_vec = lm.predict(test_X_mat, test_Y_mat)
    ```

    The model assumes that regressor variables of a given component are
    drawn from a normal distribution and the response variables are the
    sum of a linear function of the regression variables with a mean zero
    normal random variable. The `fit` method estimates the parameters of
    both normal distributions and the slope and bias of the linear model,
    separately for each component.

    Note:
    =====
    -This class emulates the API of `sklearn.mixture._base.BaseMixture`.
    -Docstrings and other documentation are not finished. Please refer
    to documentation in `sklearn.mixture._base.BaseMixture` for docstrings
    that nearly describe what is done here. Also, the docstrings in the class
    `sklearn.mixture._gaussian_mixture.GaussianMixture` may be helpful.

'''

# Standard library:
import typing
import warnings

# External packages:
import numpy as np
import scipy.special as special
import sklearn.exceptions as sklexc
import sklearn.mixture._base as base
import sklearn.utils.validation as validation

# Local modules:
import utils.utils as utils

class LinearMixture(base.BaseMixture):

    _parameter_constraints: dict = {
        **base.BaseMixture._parameter_constraints,
        'covariance_type': ['full'],
        'weights_init': ['array-like', None],
        'means_init': ['array-like', None],
        'regressor_precisions_init': ['array-like', None],
        'biases_init': ['array-like', None],
        'slopes_init': ['array-like', None],
        'response_precisions_init': ['array-like', None],
    }

    def __init__(
        self,
        n_components: int = 1,
        *,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = 'kmeans',
        weights_init: utils.array_1D = None,
        means_init: utils.array_2D = None,
        regressor_precisions_init: utils.array_3D = None,
        biases_init: utils.array_2D = None,
        slopes_init: utils.array_3D = None,
        response_precisions_init: utils.array_3D = None,
        random_state: typing.Optional[np.random.RandomState] = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
    ):

        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )

        if regressor_precisions_init is None:
            regressor_covariances_init = None
        else:
            regressor_covariances_init = np.empty(regressor_precisions_init.shape)
            for k, regressor_precision_mat in enumerate(regressor_precisions_init):
                regressor_covariances_init[k] = np.linalg.inv(regressor_precision_mat)

        if response_precisions_init is None:
            response_covariances_init = None
        else:
            response_covariances_init = np.empty(response_precisions_init.shape)
            for k, response_precision_mat in enumerate(response_precisions_init):
                response_covariances_init[k] = np.linalg.inv(response_precision_mat)

        # Fit parameter initial values:
        self._linear_model_init = utils.LinearModel(
            weight_vec=weights_init,
            gaussian_parameters=utils.GaussianParameters(
                means_init,
                regressor_covariances_init
            ),
            linear_parameters=utils.LinearParameters(
                biases_init,
                slopes_init,
                response_covariances_init,
            ),
        )

        # Fit parameters, initialized to `None`:
        self._linear_model = utils.LinearModel(
            weight_vec=None,
            gaussian_parameters=utils.GaussianParameters(),
            linear_parameters=utils.LinearParameters()
        )

        # Check that the explicit and implicit number of components match:
        if not self._linear_model_init.n_components in [None, self.n_components]:
            raise ValueError(
                f"The number of components `n_components = {self.n_components}` "
                "passed to the constructor must equal the number of components "
                f"{self._linear_model_init.n_components} implied by the initial "
                "fit parameter values also passed to the constructor, but doesn't."
            )

        # Set the number of feature (i.e. regressor) and response variables:
        self._n_features = self._linear_model_init.n_features
        self._n_responses = self._linear_model_init.n_responses

        self.n_iter_ = None
        self.lower_bound_ = None

        self._resp_init_mat = None

    @property
    def weights_init(self) -> utils.array_1D:
        return self._linear_model_init.weight_vec

    @property
    def means_init(self) -> utils.array_2D:
        return self._linear_model_init.gaussian_parameters.mean_mat

    @property
    def regressor_precisions_init(self) -> utils.array_3D:
        return self._linear_model_init.gaussian_parameters.precision_tensor

    @property
    def biases_init(self) -> utils.array_2D:
        return self._linear_model_init.linear_parameters.bias_mat

    @property
    def slopes_init(self) -> utils.array_3D:
        return self._linear_model_init.linear_parameters.slope_tensor

    @property
    def response_precisions_init(self) -> utils.array_3D:
        return self._linear_model_init.linear_parameters.precision_tensor

    @property
    def weights_(self) -> utils.array_1D:
        return self._linear_model.weight_vec

    @property
    def means_(self) -> utils.array_2D:
        return self._linear_model.gaussian_parameters.mean_mat

    @property
    def regressor_covariances_(self) -> utils.array_3D:
        return self._linear_model.gaussian_parameters.covariance_tensor

    @property
    def regressor_precisions_cholesky_(self) -> utils.array_3D:
        return self._linear_model.gaussian_parameters.precision_cholesky_tensor

    @property
    def biases_(self) -> utils.array_2D:
        return self._linear_model.linear_parameters.bias_mat

    @property
    def slopes_(self) -> utils.array_3D:
        return self._linear_model.linear_parameters.slope_tensor

    @property
    def response_covariances_(self) -> utils.array_3D:
        return self._linear_model.linear_parameters.covariance_tensor

    @property
    def response_precisions_cholesky_(self) -> utils.array_3D:
        return self._linear_model.linear_parameters.precision_cholesky_tensor

    @property
    def n_features(self) -> int:
        return self._n_features

    @n_features.setter
    def n_features(self, val: int) -> None:
        if self._linear_model_init.n_features in [None, val]:
            self._n_features = val
        else:
            raise ValueError(
                f"The number of feature variables {val} implied by the training data "
                f"must match the number of features {self._linear_model_init.n_features} "
                "implied by the initial fit parameter values passed to the constructor "
                "but doesn't."
            )

    @property
    def n_responses(self) -> int:
        return self._n_responses

    @n_responses.setter
    def n_responses(self, val: int) -> None:
        if self._linear_model_init.n_responses in [None, val]:
            self._n_responses = val
        else:
            raise ValueError(
                f"The number {val} of response variables implied by the training data "
                f"must match the number {self._linear_model_init.n_responses} of responses "
                "implied by the initial fit parameter values passed to the constructor "
                "but doesn't."
            )

    def _check_parameters(self, *args):


        '''

        Notes:
        ======
        -This method is not needed in this implementation because the checks that
        would be done here are already done during instantiation of `utils.LinearModel`.
        This method however is required by the `sklearn.mixture._base.BaseMixture`
        abstract base class, so we included it here.

        '''

        pass

    def _initialize_parameters(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
        random_state: np.random.RandomState,
    ) -> None:

        '''

        Description:
        ============
        This method treats all variables in `X_mat` concatenated with `Y_mat` as
        features and discovers clusters within this data using the base class method
        `super()._initialize_parameters`. It then fits a linear model to each cluster
        (i.e., 'component') and uses those fit parameters to seed the EM algorithm.

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).
        -The method `super()._initialize_parameters` alone does not work because it
        assumes that all variables are features and none are responses to features.
        For this reason, we overwrite it with the following method, which actually
        uses `super()._initialize_parameters` to discover clusters in the data.

        '''

        # Discover components (i.e., 'clusters') in the data:
        # Sets private attribute `self._resp_init_mat` via `self._initialize`:
        super()._initialize_parameters(np.concatenate([X_mat, Y_mat], axis=1), random_state)

        # Fit each discovered component (i.e., 'cluster') to its own linear model:
        linear_model_init = utils.fit_linear_model(X_mat, Y_mat, self._resp_init_mat)

        # Feature and response fit parameters:
        gaussian_parameters_init = linear_model_init.gaussian_parameters
        linear_parameters_init = linear_model_init.linear_parameters

        # Initialize linear model fit parameters to values passed to the constructor:
        linear_model = self._linear_model_init

        # Innitialize missing values to those found via `super()._initialize_parameters`:

        if linear_model.weight_vec is None:
            linear_model.weight_vec = np.mean(self._resp_init_mat, axis=0)

        if linear_model.gaussian_parameters.mean_mat is None:
            linear_model.gaussian_parameters.mean_mat = gaussian_parameters_init.mean_mat

        if linear_model.gaussian_parameters.covariance_tensor is None:
            linear_model.gaussian_parameters.covariance_tensor = gaussian_parameters_init.covariance_tensor

        if linear_model.linear_parameters.bias_mat is None:
            linear_model.linear_parameters.bias_mat = linear_parameters_init.bias_mat

        if linear_model.linear_parameters.slope_tensor is None:
            linear_model.linear_parameters.slope_tensor = linear_parameters_init.slope_tensor

        if linear_model.linear_parameters.covariance_tensor is None:
            linear_model.linear_parameters.covariance_tensor = linear_parameters_init.covariance_tensor

        self._set_parameters(linear_model)

    def _initialize(
        self,
        X_mat: utils.array_2D,
        resp_mat: utils.array_2D,
    ) -> None:

        '''

        Notes:
        ======
        -The argument `X_mat` is not used here. We include it to match the interface
        of `super()._initialize_parameters`.
        -We must have `resp_mat.shape[0] == self.n_components` (the number of components).
        -This method is called at the end of `super()._initialize_parameters`. It
        initializes the responsibilities to values implied by the clustering results
        from `super()._initialize_parameters`:

        '''

        self._resp_init_mat = resp_mat

    def _e_step(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
    ) -> tuple[float, utils.array_2D]:

        '''

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).
        -This code is identical to that in `base.BaseMixture._e_step`. It needs to be
        rewritten here in order to accept the response variables `Y_mat`.

        '''

        sum_comp_log_prob_xy_vec, log_resp_mat = self._estimate_log_prob_resp(X_mat, Y_mat)

        return np.mean(sum_comp_log_prob_xy_vec), log_resp_mat

    def _m_step(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
        log_resp_mat: utils.array_2D,
    ) -> None:

        '''

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[0] == log_resp_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).
        -We must have `log_resp_mat.shape[0] == self.n_components` (the number of components).

        '''

        # For each component, fit a new linear model to its data and
        # compute its weight given the new responsibilities from the 'E-step':
        linear_model = utils.fit_linear_model(X_mat, Y_mat, np.exp(log_resp_mat))
        self._set_parameters(linear_model)

    def _estimate_log_prob(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
    ) -> utils.array_2D:

        '''

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).

        '''

        # Output probability density matrix:
        log_prob_xy_mat = np.empty((X_mat.shape[0], self.n_components))

        # Compute log-probability densities for each component:
        for component_index in range(self.n_components):

            # Log-probability of regressors given the component index:
            log_prob_x_vec = utils.compute_log_gaussian_prob(
                X_mat,
                self.means_[component_index],
                self.regressor_covariances_[component_index],
            )

            # Log-probability of responses given regressors and the component index:
            log_prob_y_vec = utils.compute_log_gaussian_prob(
                Y_mat - np.dot(X_mat, self.slopes_[component_index]) - self.biases_[component_index],
                np.zeros(Y_mat.shape[1]),
                self.response_covariances_[component_index],
            )

            # Log-probability of regressors and responses given the component index:
            log_prob_xy_mat[:, component_index] = log_prob_y_vec + log_prob_x_vec

        return log_prob_xy_mat

    def _estimate_log_weights(self) -> utils.array_1D:
        return np.log(self.weights_)

    def _estimate_weighted_log_prob(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
    ) -> utils.array_2D:
        return self._estimate_log_prob(X_mat, Y_mat) + self._estimate_log_weights()

    def _estimate_log_prob_resp(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
    ) -> tuple[utils.array_1D, utils.array_2D]:

        '''

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).
        -This code is identical to that in `base.BaseMixture._estimate_log_prob_resp`.
        It needs to be rewritten here in order to accept the response variables `Y_mat`.

        '''

        # Responsibity matrix (`self.n_samples` x `self.n_components`):
        weighted_log_prob_xy_mat = self._estimate_weighted_log_prob(X_mat, Y_mat)

        # Sum responsibilities over components:
        sum_comp_log_prob_xy_vec = special.logsumexp(weighted_log_prob_xy_mat, axis=1)

        with np.errstate(under='ignore'):
            # Ignore underflow:
            log_resp_mat = weighted_log_prob_xy_mat - sum_comp_log_prob_xy_vec[:, np.newaxis]

        return sum_comp_log_prob_xy_vec, log_resp_mat

    def _compute_lower_bound(
        self,
        _,
        log_prob_norm: float
    ) -> float:
        return log_prob_norm

    def _get_parameters(self) -> utils.LinearModel:
        return self._linear_model

    def _set_parameters(
        self,
        linear_model: utils.LinearModel,
    ) -> None:
        self._linear_model = linear_model

    @base._fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
    ):

        '''

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).
        -This code is identical to that in `base.BaseMixture.fit`. It needs to be
        rewritten here in order to accept the response variables `Y_mat`.

        '''

        self.fit_predict(X_mat, Y_mat)
        return self

    def fit_predict(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
    ) -> utils.array_1D:

        '''

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).
        -This code is identical to that in `base.BaseMixture.fit_predict`. It needs
        to be rewritten here in order to accept the response variables `Y_mat`.

        '''

        # Check for consistency in the structure of the data matrices.
        # Setting `reset=True` ensures that we do not check that the number of columns
        # of `X_mat` or `Y_mat` equals the value of the inherited attribute `self.n_features_in_`:
        X_mat = self._validate_data(X_mat, dtype=[np.float64, np.float32], ensure_min_samples=2, reset=True)
        Y_mat = self._validate_data(Y_mat, dtype=[np.float64, np.float32], ensure_min_samples=2, reset=True)
        utils.check_n_samples(X_mat, Y_mat, 'regressors', 'responses')
        if X_mat.shape[0] < self.n_components:
            raise ValueError(
                'Expected n_samples >= n_components '
                f'but got n_components = {self.n_components}, '
                f'n_samples = {X_mat.shape[0]}'
            )

        # Check that the number of columns matches the number...
        self.n_features = X_mat.shape[1] # of features.
        self.n_responses = Y_mat.shape[1] # of responses.

        # NOTE: this does nothing. We include it only to match the base class code:
        self._check_parameters()

        # If we enable warm_start, we will have a unique initialization:
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = validation.check_random_state(self.random_state)

        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X_mat, Y_mat, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X_mat, Y_mat)
                    self._m_step(X_mat, Y_mat, log_resp)
                    lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

                    change = lower_bound - prev_lower_bound
                    self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                self._print_verbose_msg_init_end(lower_bound)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter

        # Should only warn about convergence if `max_iter > 0`, otherwise the
        # user is assumed to have used 0-iters initialization to get the initial means:
        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                f'Initialization {init + 1} did not converge. Try different init, '
                'parameters or increase `max_iter`, `tol` or check for degenerate data.',
                sklexc.ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # `fit_predict(X_mat, Y_mat)` are always consistent with those returned
        # by `fit(X_mat, Y_mat).predict(X_mat, Y_mat)` for any value of `max_iter`
        # and `tol` (and any `random_state`):
        _, log_resp = self._e_step(X_mat, Y_mat)

        return log_resp.argmax(axis=1)

    def predict(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
    ) -> utils.array_1D:

        '''

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).
        -The model must have been fitted by calling `self.fit` or `self.fit_predict`.
        -This code is identical to that in `base.BaseMixture.predict`. It needs to
        be rewritten here in order to accept the response variables `Y_mat`.

        '''

        # Check that the model has been fitted:
        validation.check_is_fitted(self)

        # Check for consistency in the structure of the data matrices:
        X_mat = self._validate_data(X_mat, dtype=[np.float64, np.float32], ensure_min_samples=2, reset=True)
        Y_mat = self._validate_data(Y_mat, dtype=[np.float64, np.float32], ensure_min_samples=2, reset=True)
        utils.check_n_samples(X_mat, Y_mat, 'regressors', 'responses')

        return self._estimate_weighted_log_prob(X_mat, Y_mat).argmax(axis=1)

    def predict_proba(
        self,
        X_mat: utils.array_2D,
        Y_mat: utils.array_2D,
    ) -> utils.array_2D:

        '''

        Notes:
        ======
        -We must have `X_mat.shape[0] == Y_mat.shape[0]` (the number of samples).
        -We must have `X_mat.shape[1] == self.n_features` (the number of regressors).
        -We must have `Y_mat.shape[1] == self.n_responses` (the number of responses).
        -The model must have been fitted by calling `self.fit` or `self.fit_predict`.
        -This code is identical to that in `base.BaseMixture.predict_proba`. It needs to
        be rewritten here in order to accept the response variables `Y_mat`.

        '''

        # Check that the model has been fitted:
        validation.check_is_fitted(self)

        # Check for consistency in the structure of the data matrices:
        X_mat = self._validate_data(X_mat, dtype=[np.float64, np.float32], ensure_min_samples=2, reset=True)
        Y_mat = self._validate_data(Y_mat, dtype=[np.float64, np.float32], ensure_min_samples=2, reset=True)
        utils.check_n_samples(X_mat, Y_mat, 'regressors', 'responses')

        _, log_resp_mat = self._estimate_log_prob_resp(X_mat, Y_mat)
        return np.exp(log_resp_mat)

    def sample(self, n_samples: int = 1) -> tuple[utils.array_2D, utils.array_2D]:

        '''

        Notes:
        ======
        -To be completed in the future.

        '''

        raise NotImplementedError('The method "sample" has not been implemented yet.')


if __name__ == '__main__':

    '''

    This is a smoke test. The test data comprises two distinct linear models
    with both their regressor and response variables strongly separated and
    having identical sample counts. The fit model should be ablel to recover
    the slopes and biases of these two models and their relative weights (this
    being 0.5 for either because these two models have identical sample counts).

    '''

    from test.utils.test_utils import TestData

    test_data = TestData()

    # Fit the test data to a linear mixture model:
    linear_mixture = LinearMixture(n_components=2, random_state=np.random.RandomState(1))
    linear_mixture.fit_predict(test_data.X_mat, test_data.Y_mat)

    # Check the weights of the discovered components:
    assert np.allclose(
        linear_mixture.weights_,
        np.array([0.5, 0.5]),
        rtol=1e-1,
    )

    # Check the fitted biases of the discovered components:
    assert np.allclose(
        linear_mixture.biases_,
        np.array([test_data.bias1_vec, test_data.bias2_vec]),
        rtol=1e-1,
    )

    # Check the fitted slopes of the discovered components:
    assert np.allclose(
        linear_mixture.slopes_,
        np.array([test_data.slope1_mat, test_data.slope2_mat]),
        rtol=1e-1,
    )

    # Check the means of the regressor variables of the discovered components:
    assert np.allclose(
        linear_mixture.means_,
        np.array([test_data.mean1_vec, test_data.mean2_vec]),
        rtol=1e-1,
    )

    # For covariances, don't use relative tolerance because the relative error between the estimated
    # and actual value is 1 for entries of the covariance matrix whose actual value equals 0.0:

    # Check the covariances of the regressor variables of the discovered components:
    assert np.allclose(
        linear_mixture.response_covariances_,
        np.array([test_data.res_cov1_mat, test_data.res_cov2_mat]),
        atol=1e-1,
    )

    # Check the covariances of the response variables of the discovered components:
    assert np.allclose(
        linear_mixture.regressor_covariances_,
        np.array([test_data.reg_cov1_mat, test_data.reg_cov2_mat]),
        atol=1e-1,
    )

    print('Smoke test passed!')
