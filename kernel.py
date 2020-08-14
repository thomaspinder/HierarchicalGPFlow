from gpflow.kernels import Combination
import tensorflow as tf
from copy import deepcopy


class Hierarchical(Combination):
    def __init__(self, kern_list, indicator_dims=[1]):
        Combination.__init__(self, kern_list)
        self.indicator_dims = indicator_dims

    def K(self, X, X2=None):
        K = self.kernels[0].K(X, X2)

        if X2 is None:
            X2 = deepcopy(X)

        for i, ind_dim in enumerate(self.indicator_dims):
            indX, indX2 = X[:, ind_dim : ind_dim + 1], X2[:, ind_dim : ind_dim + 1]
            mask = tf.cast(tf.equal(indX, tf.transpose(indX2)), dtype=tf.float64)
            k = self.kernels[i + 1]
            K += mask * k.K(X, X2)
        return K

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))