'''

    Module:
    =======
    Name: `example.py`
    Author: Steven Flores
    Date: 2024-02-14

    Description:
    ============
    This example script demonstrates how to use `linear_mixture.LinearMixture`
    on some test data. Running this script saves a plot to the local directory
    showing the clustering results on an illustrative example.

'''

# External packages:
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.mixture as mixture

# Local modules:
import linear_mixture

# Keep the number of regressors and responses both equal to one so this is easy to plot:
class TestData:

    def __init__(self):

        np.random.seed(1)

        # Number of regressor and response variables:
        self.n_regressors = 1
        self.n_responses = 1

        # Number of samples:
        self.n_samples = 1000

        # number of components (i.e., linear models):
        self.n_components = 2

        # Regressor parameters:
        self.mean1_vec = np.array([0])
        self.mean2_vec = np.array([0])

        assert len(self.mean1_vec) == self.n_regressors
        assert len(self.mean2_vec) == self.n_regressors

        self.reg_cov1_mat = np.array([[1]])
        self.reg_cov2_mat = np.array([[1]])

        assert self.reg_cov1_mat.shape == (self.n_regressors, self.n_regressors)
        assert self.reg_cov2_mat.shape == (self.n_regressors, self.n_regressors)

        # Linear model parameters:
        self.slope1_mat = np.array([[1]])
        self.slope2_mat = np.array([[-1]])

        assert self.slope1_mat.shape == (self.n_regressors, self.n_responses)
        assert self.slope2_mat.shape == (self.n_regressors, self.n_responses)

        self.bias1_vec = np.array([0])
        self.bias2_vec = np.array([0])

        assert len(self.bias1_vec) == self.n_responses
        assert len(self.bias2_vec) == self.n_responses

        self.res_cov1_mat = 1e-1 * np.array([[1]])
        self.res_cov2_mat = 1e-1 * np.array([[1]])

        assert self.res_cov1_mat.shape == (self.n_responses, self.n_responses)
        assert self.res_cov2_mat.shape == (self.n_responses, self.n_responses)

        self.X1_mat = np.random.multivariate_normal(self.mean1_vec, self.reg_cov1_mat, self.n_samples)
        self.X2_mat = np.random.multivariate_normal(self.mean2_vec, self.reg_cov2_mat, self.n_samples)

        # Responses without errors:
        self.Y1_mat = np.dot(self.X1_mat, self.slope1_mat) + self.bias1_vec
        self.Y2_mat = np.dot(self.X2_mat, self.slope2_mat) + self.bias2_vec

        # Responses with errors:
        self.Y1_mat += np.random.multivariate_normal(np.zeros(len(self.res_cov1_mat)), self.res_cov1_mat, self.n_samples)
        self.Y2_mat += np.random.multivariate_normal(np.zeros(len(self.res_cov1_mat)), self.res_cov2_mat, self.n_samples)

        # Component responsabilities:
        self.resp1_mat = np.stack([np.ones(self.n_samples), np.zeros(self.n_samples)], axis=1)
        self.resp2_mat = np.stack([np.zeros(self.n_samples), np.ones(self.n_samples)], axis=1)

        # Concatenate the two components:
        self.X_mat = np.concatenate([self.X1_mat, self.X2_mat], axis=0)
        self.Y_mat = np.concatenate([self.Y1_mat, self.Y2_mat], axis=0)
        self.resp_mat = np.concatenate([self.resp1_mat, self.resp2_mat], axis=0)

if __name__ == '__main__':

    test_data = TestData()

    gm = mixture.GaussianMixture(
        n_components=test_data.n_components,
        random_state=0
    ).fit(np.concatenate([test_data.X_mat, test_data.Y_mat], axis=1))
    gm_label_vec = gm.predict(np.concatenate([test_data.X_mat, test_data.Y_mat], axis=1))

    lm = linear_mixture.LinearMixture(n_components=test_data.n_components).fit(test_data.X_mat, test_data.Y_mat)
    lm_label_vec = lm.predict(test_data.X_mat, test_data.Y_mat)


    x_min = np.min(test_data.X_mat) - 1
    x_max = np.max(test_data.X_mat) + 1
    y_min = np.min(test_data.Y_mat) - 1
    y_max = np.max(test_data.Y_mat) + 1

    fig, ax_vec = plt.subplots(1, 2, figsize=(40, 10), facecolor='white')

    for ax, model_name, label_vec in zip(
        ax_vec,
        ['Gaussian Mixture Model', 'Linear Mixture Model'],
        [gm_label_vec, lm_label_vec],
    ):

        for component, color in enumerate(mcolors.TABLEAU_COLORS):
            if component >= test_data.n_components:
                break
            ax.scatter(
                test_data.X_mat[label_vec == component],
                test_data.Y_mat[label_vec == component],
                alpha=0.3,
                c=color,
                label=f'component {component}',
            )

        ax.legend(fontsize=15)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(labelsize=15)

        ax.set_title(model_name, fontsize=20)

        ax.set_xlabel('x', fontsize=15)
        ax.set_ylabel('y', fontsize=15)

    plt.savefig('example.png', )
