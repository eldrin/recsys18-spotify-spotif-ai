from scipy import sparse as sp
import pandas as pd
import numpy as np
import torch


TORCH_DTYPE_TENSOR = {
    'float': torch.FloatTensor,
    'int': torch.LongTensor
}


def numpy2torchvar(ndarray, dtype='float', gpu=True):
    tensor = TORCH_DTYPE_TENSOR[dtype](ndarray)
    if gpu:
        return torch.autograd.Variable(tensor.cuda())
    else:
        return torch.autograd.Variable(tensor)


def read_data(fn, delimiter=',', f_cast=int, shape=None):
    """"""
    ll = pd.read_csv(fn, sep=delimiter, header=None, index_col=None)
    i = ll[0]
    j = ll[1]

    if len(ll.columns) > 2:
        v = ll[2]
        ll.columns = ['playlist', 'track', 'value']
    else:
        v = [1] * len(i)
        ll.columns = ['playlist', 'track']

    if shape is None:
        shape = (max(i) + 1, max(j) + 1)

    D = sp.coo_matrix((v, (i, j)), shape=shape).tocsr()
    return D, ll


def sparse2triplet(S):
    """"""
    return np.array(sp.find(S)).T


def negative_sampling(X):
    """"""
    pass


