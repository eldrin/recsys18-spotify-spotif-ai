import os
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from implicit.als import AlternatingLeastSquares
from data import SeqTensor


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


class UserRNN(nn.Module):
    """"""
    def __init__(self, n_components, n_users, n_hid=100, n_out=16,
                 user_train=True, n_layers=1, drop_out=0, sparse_embedding=True):
        """"""
        super(UserRNN, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_layers = n_layers
        self.n_out = n_out
        self.is_cuda = False

        # setup learnable embedding layers
        self.emb = nn.Embedding(
            n_users, n_components, sparse=sparse_embedding)
        self.emb.weight.requires_grad = user_train
        self.user_rnn = nn.LSTM(n_components, n_hid, n_layers,
                                batch_first=True, dropout=drop_out)
        self.user_out = nn.Linear(n_hid, n_out)

    def forward(self, pid):
        """
        pid: SeqTensor instance for batch of playlist
        """
        # process seqs
        pid = SeqTensor(pid, None, None, is_gpu=self.is_cuda)

        # process rnn
        emb_pl = self.emb(Variable(pid.seq))
        emb_pl = pack_padded_sequence(emb_pl, pid.lengths.tolist(), batch_first=True)
        out_u, hid_u = self.user_rnn(emb_pl)

        # unpack & unsort batch order
        hid_u = pid.unsort(hid_u[0][-1])  # only take last rnn layer

        # obtain final estimation
        out_u = self.user_out(hid_u)
        return out_u

    def user_factor(self, pid):
        """"""
        pid = SeqTensor(pid, None, None, is_gpu=self.is_cuda)
        emb_pl = self.emb(Variable(pid.seq))
        emb_pl = pack_padded_sequence(emb_pl, pid.lengths.tolist(), batch_first=True)
        out_u, hid_u = self.user_rnn(emb_pl)

        out_u = self.user_out(pid.unsort(hid_u[0][-1]))
        return out_u
