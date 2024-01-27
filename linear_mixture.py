

import numpy as np
import nptyping as npt
import scipy.special as special
import sklearn.cluster as cluster
import sklearn.mixture as mixture
import sklearn.utils.validation as validation

import utils.utils as utils


class LinearMixture(mixture._base.BaseMixture):

    def __init__(
        self,
        n_components=1,
        *,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
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

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init

        self.weights_ = None
        self.biases_ = None
        self.means_ = None
        self.slopes_ = None
        self.noise_ = None
        self.covariances_ = None

        self.n_iter_ = None
        self.lower_bound_ = None
        self.converged_ = False

    @property
    def n_features(self) -> int:
        if self.slopes_ is None:
            return None
        return self.slopes_.shape(0)

    @property
    def n_responses(self) -> int:
        if self.slopes_ is None:
            return None
        return self.slopes_.shape(1)


    def _initialize(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        random_state: int,
    ) -> None:

        n_samples = X_mat.shape[0]

        assert Y_mat.shape[0] == n_samples

        resp_mat = np.zeros((n_samples, self.n_components))
        label = (
            cluster.KMeans(
                n_clusters=self.n_components, n_init=1, random_state=random_state
            )
            .fit(np.concatenate([X_mat, Y_mat], axis=1))
            .labels_
        )
        resp_mat[np.arange(n_samples), label] = 1

        linear_model = utils.fit_linear_model(X_mat, Y_mat, resp_mat)

        self.weights_ = np.mean(resp_mat, axis=0)
        self.biases_ = linear_model.linear_parameters.bias_mat
        self.means_ = linear_model.gaussian_parameters.mean_mat
        self.slopes_ = linear_model.linear_parameters.slope_tensor
        self.noise_ = linear_model.linear_parameters.covariance_tensor
        self.covariances_ = linear_model.gaussian_parameters.covariance_tensor

    def _estimate_log_prob(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> npt.NDArray[npt.Shape['*, *'], npt.Float]:

        n_samples = X_mat.shape[0]
        n_responses = Y_mat.shape[1]

        # Check that the response and regressor have the same number of samples:
        assert Y_mat.shape[0] == n_samples

        # Output probability density matrix:
        log_prob_xy_mat = np.empty((n_samples, self.n_components))

        for component_index in range(self.n_components):

            # Log-probability of regressors given the component index:
            log_prob_x = utils.compute_log_prob_gaussian(
                X_mat,
                self.means_[component_index],
                self.covariances_[component_index],
            )

            # Log-probability of responses given regressors and the component index:
            log_prob_y = utils.compute_log_prob_gaussian(
                Y_mat - np.dot(X_mat, self.slopes_[component_index]) - self.biases_[component_index],
                np.zeros(n_responses),
                self.noise_[component_index],
            )

            # Log-probability of regressors and responses given the component index:
            log_prob_xy_mat[:, component_index] = log_prob_y + log_prob_x

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

        # Check that the response and regressor have the same number of samples:
        assert Y_mat.shape[0] == X_mat.shape[0]

        # TODO: other data consistency checks, as implied by the shape of the bias and slope vectors/matrices?

        # Responsibity matrix (`self.n_samples` x `self.n_components`):
        weighted_log_prob_xy_mat = self._estimate_weighted_log_prob(X_mat, Y_mat)

        # Sum responsibilities over components:
        sum_comp_log_prob_xy_vec = special.logsumexp(weighted_log_prob_xy_mat, axis=1)

        with np.errstate(under='ignore'):
            # Ignore underflow:
            log_resp_mat = weighted_log_prob_xy_mat - sum_comp_log_prob_xy_vec[:, np.newaxis]

        # TODO: why return the first argument? What we really want is `weighted_log_prob_xy_mat`
        # averaged over the component index with weights equal to the responsibilities. But why
        # do that here? It would only be done here to match the interface for scikit-learn.
        return sum_comp_log_prob_xy_vec, log_resp_mat

    def _e_step(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> tuple[float, npt.NDArray[npt.Shape['*, *'], npt.Float]]:

        # Check that the response and regressor have the same number of samples:
        assert Y_mat.shape[0] == X_mat.shape[0]

        sum_comp_log_prob_xy_vec, log_resp_mat = self._estimate_log_prob_resp(X_mat, Y_mat)
        # TODO: the mean here seems to be a proxy for the true loss function that we are
        # maximizing. Why do that?
        return np.mean(sum_comp_log_prob_xy_vec), log_resp_mat

    def _m_step(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        log_resp_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ) -> None:

        n_samples = X_mat.shape[0]

        # Check that the response and regressor have the same number of samples:
        assert Y_mat.shape[0] == X_mat.shape[0]

        # Check the shape of the matrix of responsibilities:
        assert log_resp_mat.shape == (n_samples, self.n_components)

        # Responsibilities:
        resp_mat = np.exp(log_resp_mat)

        linear_model = utils.fit_linear_model(X_mat, Y_mat, resp_mat)

        self.weights_ = linear_model.weight_vec
        self.biases_ = linear_model.linear_parameters.bias_mat
        self.means_ = linear_model.gaussian_parameters.mean_mat
        self.slopes_ = linear_model.linear_parameters.slope_tensor
        self.noise_ = linear_model.linear_parameters.covariance_tensor
        self.covariances_ = linear_model.gaussian_parameters.covariance_tensor

    def _check_parameters(self, X):
        """Check the Gaussian mixture parameters are well defined."""
        # TODO: update this to check the mean and covariance of X and the noise of Y:
        _, n_features = X.shape

        if self.weights_init is not None:
            self.weights_init = mixture._gaussian_mixture._check_weights(
                self.weights_init, self.n_components
            )

        if self.means_init is not None:
            self.means_init = mixture._gaussian_mixture._check_means(
                self.means_init, self.n_components, n_features
            )

        if self.precisions_init is not None:
            self.precisions_init = mixture._gaussian_mixture._check_precisions(
                self.precisions_init,
                self.covariance_type,
                self.n_components,
                n_features,
            )

    def fit(self, X, Y):
        self.fit_predict(X, Y)
        return self

    # NOTE: we must rewrite this method from what is in scikit-learn
    # because we need to use Y. Therefore, we can make this whatever we want.
    # Most of what we have will be based off scikit-learn. Acknowlede their code.
    def fit_predict(self, X, Y):

        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)

        # # if we enable warm_start, we will have a unique initialisation
        # do_init = not (self.warm_start and hasattr(self, "converged_"))
        # n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = validation.check_random_state(self.random_state)

        # SF:
        n_init = self.n_init
        random_state = 1
        do_init = True

        n_samples, _ = X.shape # TODO: from the original code, but not used there...
        for init in range(n_init):
            # TODO: bring back...
            # self._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize(X, Y, random_state)

            lower_bound = -np.inf if do_init else self.lower_bound_

            if self.max_iter == 0:
                best_params = self._get_parameters()
                best_n_iter = 0
            else:
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound

                    log_prob_norm, log_resp = self._e_step(X, Y)
                    self._m_step(X, Y, log_resp)
                    # TODO: this function is trivial for the Gaussian mixture model...
                    # lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)
                    lower_bound = log_prob_norm

                    change = lower_bound - prev_lower_bound
                    # TODO: bring back
                    # self._print_verbose_msg_iter_end(n_iter, change)

                    if abs(change) < self.tol:
                        self.converged_ = True
                        break

                # TODO: bring back...
                # self._print_verbose_msg_init_end(lower_bound)

                if lower_bound > max_lower_bound or max_lower_bound == -np.inf:
                    max_lower_bound = lower_bound
                    best_params = self._get_parameters()
                    best_n_iter = n_iter

        # # Should only warn about convergence if max_iter > 0, otherwise
        # # the user is assumed to have used 0-iters initialization
        # # to get the initial means.
        # if not self.converged_ and self.max_iter > 0:
        #     warnings.warn(
        #         "Initialization %d did not converge. "
        #         "Try different init parameters, "
        #         "or increase max_iter, tol "
        #         "or check for degenerate data." % (init + 1),
        #         ConvergenceWarning,
        #     )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X, Y)

        return log_resp.argmax(axis=1)

    def predict(
        self,
        X_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
        Y_mat: npt.NDArray[npt.Shape['*, *'], npt.Float],
    ):
        return self._estimate_weighted_log_prob(X_mat, Y_mat).argmax(axis=1)

    def _get_parameters(self):
        return (
            self.weights_,
            self.biases_,
            self.means_,
            self.slopes_,
            self.noise_,
            self.covariances_,
        )

    def _set_parameters(self, params):
        (
            self.weights_,
            self.biases_,
            self.means_,
            self.slopes_,
            self.noise_,
            self.covariances_,
        ) = params

if __name__ == '__main__':

    from test.utils.test_utils import TestData

    test_data = TestData()

    linear_mixture = LinearMixture(n_components=2)
    linear_mixture.fit_predict(test_data.X_mat, test_data.Y_mat)
    print('weights:', linear_mixture.weights_)
    print('biases:', linear_mixture.biases_)
    print('means:', linear_mixture.means_)
    print('slopes:', linear_mixture.slopes_)
    print('noise:', linear_mixture.noise_)
    print('covariances:', linear_mixture.covariances_)

    assert np.allclose(linear_mixture.weights_, np.array([0.5, 0.5]), rtol=1e-1)
    assert np.allclose(linear_mixture.biases_, np.array([test_data.bias1_vec, test_data.bias2_vec]), rtol=1e-1)
    assert np.allclose(linear_mixture.means_, np.array([test_data.mean1_vec, test_data.mean2_vec]), rtol=1e-1)
    assert np.allclose(linear_mixture.slopes_, np.array([test_data.slope1_mat, test_data.slope2_mat]), rtol=1e-1)
    assert np.allclose(linear_mixture.noise_, np.array([test_data.res_cov1_mat, test_data.res_cov2_mat]), atol=1e-1)
    # Don't use relative tolerance because the relative error between the estimated and
    # actual value is 1 for entries of the covariance matrix whose actual value equals 0.0:
    assert np.allclose(linear_mixture.covariances_, np.array([test_data.reg_cov1_mat, test_data.reg_cov2_mat]), atol=1e-1)