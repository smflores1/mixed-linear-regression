

import warnings
import typing
import numpy as np
import nptyping as npt
import scipy.special as special
import sklearn.mixture._base as base
import sklearn.utils.validation as validation
import sklearn.exceptions as sklexc

import utils.utils as utils



class LinearMixture(base.BaseMixture):

    # TODO: type hinting throughout...
    def __init__(
        self,
        n_components=1,
        *,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        regressor_precisions_init=None,
        biases_init=None,
        slopes_init=None,
        response_precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
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

        # For each component, we estimate a separate covariance matrix
        # for both the regressor variables and the response variables:
        self.covariance_type = 'full'

        # Maybe run _check_parameters with no data X to see if these inputs are ok...

        self.weights_init = weights_init

        self.means_init = means_init
        self.regressor_precisions_init = regressor_precisions_init

        self.biases_init = biases_init
        self.slopes_init = slopes_init
        self.response_precisions_init = response_precisions_init

        self.weights_ = None

        self.means_ = None
        # TODO: make into a property:
        self.regressor_covariance_ = None
        self.regressor_precisions_ = None
        self.regressor_precisions_cholesky_ = None

        self.biases_ = None
        self.slopes_ = None
        self.response_covariance_ = None
        self.response_precisions_cholesky_ = None

        self.n_iter_ = None
        self.lower_bound_ = None

        self._resp_init_mat = None

    @property
    def n_features(self) -> int:
        if self.slopes_init is None:
            return None
        return self.slopes_init.shape[0]

    @property
    def n_responses(self) -> int:
        if self.slopes_init is None:
            return None
        return self.slopes_init.shape[1]

    @property
    def regressor_precisions(self) -> typing.Optional[npt.NDArray[npt.Shape['*, *, *'], npt.Float]]:
        if self.regressor_precisions_cholesky_ is None:
            return None
        regressor_precisions_tensor = np.empty(self.regressor_precisions_cholesky_.shape)
        for component_index, cholesky_mat in enumerate(self.regressor_precisions_cholesky_):
            regressor_precisions_tensor[component_index] = np.dot(cholesky_mat, cholesky_mat.T)
        return regressor_precisions_tensor

    @property
    def response_precisions(self) -> typing.Optional[npt.NDArray[npt.Shape['*, *, *'], npt.Float]]:
        if self.response_precisions_cholesky_ is None:
            return None
        response_precisions_tensor = np.empty(self.response_precisions_cholesky_.shape)
        for component_index, cholesky_mat in enumerate(self.response_precisions_cholesky_):
            response_precisions_tensor[component_index] = np.dot(cholesky_mat, cholesky_mat.T)
        return response_precisions_tensor

    def _check_parameters(self, X, Y):
        """Check the Gaussian mixture parameters are well defined."""
        # TODO: update this to check the mean and covariance of X and the noise of Y:

        # TODO: write a method to check that the number of samples between X, Y, and the responsibility matrix matches...

        # TODO: write a method to check that the responsibility matrix has self.n_components columns.

        _, n_features = X.shape
        _, n_responses = Y.shape

        if self.weights_init is not None:
            self.weights_init = utils.check_weights(
                self.weights_init, self.n_components
            )

        if self.means_init is not None:
            self.means_init = utils.check_means(
                self.means_init, self.n_components, n_features
            )

        # TODO: Check biases?
        # TODO: Check slopes?

        if self.biases_init is not None:
            self.biases_init = utils.check_biases(
                self.biases_init, self.n_components, n_responses
            )
        if self.slopes_init is not None:
            self.slopes_init = utils.check_slopes(
                self.slopes_init, self.n_components, n_features, n_responses
            )

        # TODO: do we check self.regressor_covaraince and self.response_covariance?

        if self.regressor_precisions_init is not None:
            self.regressor_precisions_init = utils.check_precisions(
                self.regressor_precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

        if self.response_precisions_init is not None:
            self.response_precisions_init = utils.check_precisions(
                self.response_precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def _initialize_parameters(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        random_state: np.random.RandomState,
    ) -> None:

        # Set's private attribute `self._resp_init_mat`:
        super()._initialize_parameters(
            np.concatenate([X_mat, Y_mat], axis=1),
            random_state,
        )


        # TODO: check the types and dimensions before setting these?
        # Should we make getters and setters out of these to enforce this?
        # It's probably easier to call a single method that checks everything
        # whenever these parameters are set.

        # Comment that we have to do this code here because _initialize as it is used
        # in super()._initialize_parameters does not let us split X_mat and Y_mat.
        linear_model = utils.fit_linear_model(X_mat, Y_mat, self._resp_init_mat)

        # TODO: can we use _set_parameters for this?

        if self.weights_init is None:
            self.weights_ = np.mean(self._resp_init_mat, axis=0)
        else:
            self.weights_ = self.weights_init

        if self.means_init is None:
            self.means_ = linear_model.gaussian_parameters.mean_mat
        else:
            self.means_ = self.means_init

        if self.biases_init is None:
            self.biases_ = linear_model.linear_parameters.bias_mat
        else:
            self.biases_ = self.biases_init

        if self.slopes_init is None:
            self.slopes_ = linear_model.linear_parameters.slope_tensor
        else:
            self.slopes_ = self.slopes_init

        if self.regressor_precisions_init is None:
            self.regressor_covariance_ = linear_model.gaussian_parameters.covariance_tensor
            self.regressor_precisions_cholesky_ = utils.compute_precision_cholesky(
                self.regressor_covariance_, self.covariance_type
            )
        else:
            self.regressor_precisions_cholesky_ = utils.compute_precision_cholesky_from_precisions(
                self.regressor_precisions_init, self.covariance_type
            )

        if self.regressor_precisions_init is None:
            self.regressor_covariance_ = linear_model.gaussian_parameters.covariance_tensor
            self.regressor_precisions_cholesky_ = utils.compute_precision_cholesky(
                self.regressor_covariance_, self.covariance_type
            )
        else:
            self.regressor_precisions_cholesky_ = utils.compute_precision_cholesky_from_precisions(
                self.regressor_precisions_init, self.covariance_type
            )

        if self.response_precisions_init is None:
            self.response_covariance_ = linear_model.linear_parameters.covariance_tensor
            self.response_precisions_cholesky_ = utils.compute_precision_cholesky(
                self.response_covariance_, self.covariance_type
            )
        else:
            self.response_precisions_cholesky_ = utils.compute_precision_cholesky_from_precisions(
                self.response_precisions_init, self.covariance_type
            )

        self.response_covariance_ = linear_model.linear_parameters.covariance_tensor
        self.regressor_covariance_ = linear_model.gaussian_parameters.covariance_tensor

    # TODO: document that this is a weird thing to do and why...
    def _initialize(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> None:
        self._resp_init_mat = resp_mat

    def _e_step(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> tuple[float, npt.NDArray[npt.Shape['*, *'], npt.Float]]:

        utils.check_n_samples(X_mat, Y_mat, 'regressors', 'responses')

        sum_comp_log_prob_xy_vec, log_resp_mat = self._estimate_log_prob_resp(X_mat, Y_mat)

        return np.mean(sum_comp_log_prob_xy_vec), log_resp_mat

    def _m_step(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        log_resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> None:

        utils.check_n_samples(X_mat, Y_mat, 'regressors', 'responses')
        utils.check_n_samples(X_mat, log_resp_mat, 'regressors', 'responsibilities')

        utils.check_shape(log_resp_mat, (X_mat.shape[0], self.n_components), 'responsibilities')

        linear_model = utils.fit_linear_model(X_mat, Y_mat, np.exp(log_resp_mat))

        # TODO: why not do this through _set_parameters()?
        self.weights_ = linear_model.weight_vec
        self.biases_ = linear_model.linear_parameters.bias_mat
        self.means_ = linear_model.gaussian_parameters.mean_mat
        self.slopes_ = linear_model.linear_parameters.slope_tensor
        self.regressor_covariance_ = linear_model.gaussian_parameters.covariance_tensor
        self.response_covariance_ = linear_model.linear_parameters.covariance_tensor
        self.regressor_precisions_cholesky_ = utils.compute_precision_cholesky(
            self.regressor_covariance_, self.covariance_type
        )
        self.response_precisions_cholesky_ = utils.compute_precision_cholesky(
            self.response_covariance_, self.covariance_type
        )

    def _estimate_log_prob(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> npt.NDArray[npt.Shape['*, *'], npt.Float]:

        utils.check_n_samples(X_mat, Y_mat, 'regressors', 'responses')

        # TODO: check that X_mat has the right shape relative to the mean and covariances...
        # TODO: check that Y_mat has the right shape...

        # Output probability density matrix:
        log_prob_xy_mat = np.empty((X_mat.shape[0], self.n_components))

        # TODO: compute the cholesky decomponsition of the precision matrix...

        for component_index in range(self.n_components):

            # Log-probability of regressors given the component index:
            log_prob_x_vec = utils.compute_log_gaussian_prob(
                X_mat,
                self.means_[component_index],
                self.regressor_covariance_[component_index],
            )

            # Log-probability of responses given regressors and the component index:
            log_prob_y_vec = utils.compute_log_gaussian_prob(
                Y_mat - np.dot(X_mat, self.slopes_[component_index]) - self.biases_[component_index],
                np.zeros(Y_mat.shape[1]),
                self.response_covariance_[component_index],
            )

            # Log-probability of regressors and responses given the component index:
            log_prob_xy_mat[:, component_index] = log_prob_y_vec + log_prob_x_vec

        return log_prob_xy_mat

    def _estimate_log_weights(self) -> npt.NDArray[npt.Shape['*'], npt.Float]:
        return np.log(self.weights_)

    def _estimate_weighted_log_prob(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> npt.NDArray[npt.Shape['*, *'], npt.Float]:
        return self._estimate_log_prob(X_mat, Y_mat) + self._estimate_log_weights()

    # TODO: somehow, the dimensionality of the data is encoded in the dimensions of the parameters.
    # But I thought that LinearMixture was supposed to be agnostic about the data?
    def _estimate_log_prob_resp(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> tuple[npt.NDArray[npt.Shape['*'], npt.Float], npt.NDArray[npt.Shape['*, *'], npt.Float]]:

        utils.check_n_samples(X_mat, Y_mat, 'regressors', 'responses')

        # TODO: other data consistency checks, as implied by the shape of the bias and slope vectors/matrices?

        # Responsibity matrix (`self.n_samples` x `self.n_components`):
        weighted_log_prob_xy_mat = self._estimate_weighted_log_prob(X_mat, Y_mat)

        # Sum responsibilities over components:
        sum_comp_log_prob_xy_vec = special.logsumexp(weighted_log_prob_xy_mat, axis=1)

        with np.errstate(under='ignore'):
            # Ignore underflow:
            log_resp_mat = weighted_log_prob_xy_mat - sum_comp_log_prob_xy_vec[:, np.newaxis]

        return sum_comp_log_prob_xy_vec, log_resp_mat

    # TODO: clean up the function signature:
    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm


    # TODO: Have this return a linear model object...
    def _get_parameters(self):
        return(
            self.weights_,
            self.means_,
            self.regressor_covariance_,
            self.regressor_precisions_cholesky_,
            self.biases_,
            self.slopes_,
            self.response_covariance_,
            self.response_precisions_cholesky_,
        )

    # TODO: make this accept a linear model as an argument...
    def _set_parameters(self, params):

        (
            weights_,
            means_,
            regressor_covariance_,
            regressor_precisions_cholesky_,
            biases_,
            slopes_,
            response_covariance_,
            response_precisions_cholesky_,
        ) = params

        if self.n_features is None:
            n_features = self.means_.shape[1]
        else:
            n_features = self.n_features

        if self.n_responses is None:
            n_responses = self.biases_.shape[1]
        else:
            n_responses = self.n_responses

        weights_ = utils.check_weights(weights_, self.n_components)

        means_ = utils.check_means(means_, self.n_components, n_features)

        regressor_precisions_cholesky_ = utils.check_precisions_cholesky(
            regressor_precisions_cholesky_,
            self.n_components,
            n_features,
        )

        biases_ = utils.check_biases(biases_, self.n_components, n_responses)
        slopes_ = utils.check_slopes(slopes_, self.n_components, n_features, n_responses)

        response_precisions_cholesky_ = utils.check_precisions_cholesky(
            response_precisions_cholesky_,
            self.n_components,
            n_responses,
        )

        self.weights_ = weights_
        self.biases_ = biases_
        self.means_ = means_
        self.slopes_ = slopes_
        self.regressor_covariance_ = regressor_covariance_
        self.response_covariance_ = response_covariance_
        self.regressor_precisions_cholesky_ = regressor_precisions_cholesky_
        self.response_precisions_cholesky_ = response_precisions_cholesky_


    def fit(self, X, Y):
        self.fit_predict(X, Y)
        return self

    # NOTE: we must rewrite this method from what is in scikit-learn
    # because we need to use Y. Therefore, we can make this whatever we want.
    # Most of what we have will be based off scikit-learn. Acknowlede their code.
    def fit_predict(self, X, Y):

        # TODO: check that X has the right number of features if this is frozen by
        # usage of init variables. Otherwise don't.

        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                'Expected n_samples >= n_components '
                f'but got n_components = {self.n_components}, '
                f'n_samples = {X.shape[0]}'
            )
        # TODO: validate data in Y too...
        self._check_parameters(X, Y)

        # If we enable warm_start, we will have a unique initialization:
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = validation.check_random_state(self.random_state)

        for init in range(n_init):
            self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, Y, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X, Y)
                    self._m_step(X, Y, log_resp)
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

        # Should only warn about convergence if max_iter > 0, otherwise
        # the user is assumed to have used 0-iters initialization
        # to get the initial means.
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
        # `fit_predict(X)` are always consistent with fit(X).predict(X)
        # for any value of `max_iter` and `tol` (and any `random_state`):
        _, log_resp = self._e_step(X, Y)

        return log_resp.argmax(axis=1)

    def predict(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ):
        return self._estimate_weighted_log_prob(X_mat, Y_mat).argmax(axis=1)


if __name__ == '__main__':

    from test.utils.test_utils import TestData

    test_data = TestData()

    linear_mixture = LinearMixture(n_components=2, random_state=np.random.RandomState(1))
    linear_mixture.fit_predict(test_data.X_mat, test_data.Y_mat)
    print('weights:', linear_mixture.weights_)
    print('biases:', linear_mixture.biases_)
    print('means:', linear_mixture.means_)
    print('slopes:', linear_mixture.slopes_)
    print('rseponse covariance:', linear_mixture.response_covariance_)
    print('regressor covariance:', linear_mixture.regressor_covariance_)

    assert np.allclose(linear_mixture.weights_, np.array([0.5, 0.5]), rtol=1e-1)
    assert np.allclose(linear_mixture.biases_, np.array([test_data.bias1_vec, test_data.bias2_vec]), rtol=1e-1)
    assert np.allclose(linear_mixture.means_, np.array([test_data.mean1_vec, test_data.mean2_vec]), rtol=1e-1)
    assert np.allclose(linear_mixture.slopes_, np.array([test_data.slope1_mat, test_data.slope2_mat]), rtol=1e-1)
    assert np.allclose(linear_mixture.response_covariance_, np.array([test_data.res_cov1_mat, test_data.res_cov2_mat]), atol=1e-1)
    # Don't use relative tolerance because the relative error between the estimated and
    # actual value is 1 for entries of the covariance matrix whose actual value equals 0.0:
    assert np.allclose(linear_mixture.regressor_covariance_, np.array([test_data.reg_cov1_mat, test_data.reg_cov2_mat]), atol=1e-1)