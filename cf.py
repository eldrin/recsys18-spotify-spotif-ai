import os
import numpy as np
import implicit
from implicit.als import AlternatingLeastSquares


class ImplicitALS(AlternatingLeastSquares):
    """ Simple sub-class for `implicit`s ALS algorithm """
    def __init__(self, n_components, regularization=1e-3, alpha=100,
                 n_iters=15, dtype=np.float32, use_gpu=False):
        """
        Args:
            n_components (int): n_factors
            regularization (float): regularization term weight
            alpha (float): confidence coefficient
        """
        super(ImplicitALS, self).__init__(
            factors=n_components, regularization=regularization,
            use_gpu=use_gpu, iterations=n_iters, dtype=np.float32
        )
	self.alpha = alpha

    def fit(self, X):
	os.environ['OPENBLAS_NUM_THREADS'] = '1'
        X.data = X.data * (self.alpha - 1.)
        super(ImplicitALS, self).fit(X.T)
	os.environ['OPENBLAS_NUM_THREADS'] = '8'

    def predict_k(self, u, k=500):
        """"""
        if not hasattr(self, 'user_factors') or not hasattr(self, 'item_factors'):
            raise ValueError('[Error] model first should be fit!')
        r = -self.user_factors[u].dot(self.item_factors.T)
        ix = np.argpartition(r, k)[:k]
        return ix[r[ix].argsort()]
