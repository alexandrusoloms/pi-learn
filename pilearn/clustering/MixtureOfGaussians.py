import numpy as np
from matplotlib import pyplot as plt


class MoG(object):
    def __init__(self, X, k, n_iter=100, verbose=False, covariance=False):
        """

        :param X: data
        :param k: number of clusters
        :param n_iter: number of iteration (default: 100)
        :param verbose: <bool> set this to true to see iterations (default: False)
        :param covariance: <bool> set this to true to use a full covariance matrix (default: False)
        """
        self.X = X
        self.k = k
        self.n_iter = n_iter
        self.verbose = verbose
        self.covariance = covariance
        self.n_samples, self.n_features = self.X.shape
        self.p = np.ones(shape=(1, self.k)) / self.k

        self.s2 = np.zeros(shape=(self.n_features, self.n_features, self.k))
        self.Z = np.zeros(shape=(self.n_samples, self.k))
        self.mu = np.zeros(shape=(self.n_features, self.k))
        self.__init_covariance_centroids()

    def __init_covariance_centroids(self):
        """
        initializes the covariance and centroid ( or mean) values
        :return:
        """
        self.mu = self.X[[int(i) for i in np.ceil(self.n_samples * np.random.rand(self.k)).tolist()]]

        for i in range(self.k):
            self.s2[:, :, i] = np.cov(self.X.T) / self.k

    def train(self):
        """

        :return:
        """
        for t in range(self.n_iter):
            if self.verbose:
                if t % 5 == 0:
                    print('t={}'.format(t))

            # run EM for niter iterations
            for i in range(self.k):
                sub_op = self.X - np.repeat([self.mu.T[:, i]], repeats=self.n_samples, axis=0).reshape(self.X.shape)
                self.Z[:, i] = self.p.T[i] * np.linalg.det(self.s2[:, :, i]) ** (-.5) * \
                               np.exp((-.5) * np.sum(sub_op.dot(np.linalg.inv(self.s2[:, :, i])) * sub_op, axis=1))

            self.Z = self.Z / np.repeat(np.sum(self.Z, axis=1).reshape(-1, 1), repeats=self.k).reshape(self.Z.shape)
            # do the M-step
            for i in range(self.k):
                self.mu.T[:, i] = self.X.T.dot(self.Z[:, i]) / np.sum(self.Z[:, i])

                sub_op = self.X - np.repeat([self.mu.T[:, i]], repeats=self.n_samples, axis=0).reshape(self.X.shape)

                if self.covariance:
                    self.s2[:, :, i] = ((sub_op *
                                         np.repeat(self.Z[:, i], repeats=1, axis=0).reshape(-1, 1)).T.dot(sub_op)) / \
                                       sum(self.Z[:, i]) + np.eye(self.n_features) * .00001
                # we will fit Gaussian with diagonal covariances
                else:
                    self.s2[:, :, i] = np.diag(
                        np.divide((sub_op ** 2).T.dot(self.Z[:, i]), sum(self.Z[:, i]))
                    )
                self.p.T[i] = np.mean(self.Z[:, i])

    def plot(self, title, figs_path, n=11):

        # palette = ['r', 'b', 'y', 'm', 'c', '']
        # plt.scatter(self.X[:, 0], self.X[:, 1], alpha=.7)
        for i in range(self.k):

            theta = np.array(list(range(n))) / (n - 1) * 2 * np.pi
            e_points = np.sqrt((2*self.s2[:, :, i])).dot(np.array([np.cos(theta), np.sin(theta)])) + \
                       self.mu.T[:, i].reshape(-1, 1) * np.ones(shape=(1, n))
            plt.plot(e_points[0, :], e_points[1, :])
            plt.plot(self.mu.T[:, i][0], self.mu.T[:, i][1], 'xk')
        if title:
            plt.title(title)
        plt.savefig(figs_path)
        plt.show()
