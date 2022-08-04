"""
This module defines the class for Singular Value Decomposition related
 operations.
"""
import numpy as np

class SingularValueDecomposition:
    """
    Class to perform various calculations based on Sigular Value Decomposition.
    """

    def decompose(self, power_signals_d, rank_k=4):
        """
        Arguments
        ---------
        power_signals_d : numpy array
            Representing a matrix with row for dates and column for time of day,
            containing input power signals.

        Keyword arguments
        -----------------
        rank_k : integer
            Rank of the resulting low rank matrices.
        """

        (left_singular_vectors_u, singular_values_sigma,
         right_singular_vectors_v) = np.linalg.svd(power_signals_d)
        left_singular_vectors_u, right_singular_vectors_v = \
            self._adjust_singular_vectors(left_singular_vectors_u,
                                           right_singular_vectors_v)
        self._left_singular_vectors_u = left_singular_vectors_u
        self._singular_values_sigma = singular_values_sigma
        self._right_singular_vectors_v = right_singular_vectors_v

        self._matrix_l0 = self._left_singular_vectors_u[:, :rank_k]
        self._matrix_r0 = np.diag(self._singular_values_sigma[:rank_k]).dot(
            right_singular_vectors_v[:rank_k, :])

    def _adjust_singular_vectors(self, left_singular_vectors_u,
                                  right_singular_vectors_v):

        if np.sum(left_singular_vectors_u[:, 0]) < 0:
            left_singular_vectors_u[:, 0] *= -1
            right_singular_vectors_v[0] *= -1

        return left_singular_vectors_u, right_singular_vectors_v

    @property
    def left_singular_vectors_u(self):
        return self._left_singular_vectors_u

    @property
    def singular_values_sigma(self):
        return self._singular_values_sigma

    @property
    def right_singular_vectors_v(self):
        return self._right_singular_vectors_v

    @property
    def matrix_l0(self):
        return self._matrix_l0

    @property
    def matrix_r0(self):
        return self._matrix_r0
