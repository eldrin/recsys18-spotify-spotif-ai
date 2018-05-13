from scipy import sparse as sp
import pandas as pd
import numpy as np


def read_data(fn, delimiter=',', f_cast=int, shape=None):
    """"""
    ll = pd.read_csv(fn, sep=delimiter, header=None, index_col=None)
    # offset_i = ll[0].min()
    # offset_j = ll[1].min()
    # i = ll[0] - offset_i
    # j = ll[1] - offset_j
    i = ll[0]
    j = ll[1]

    if len(ll.columns) > 2:
        v = ll[2]
    else:
        v = [1] * len(i)

    if shape is None:
        shape = (max(i) + 1, max(j) + 1)

    D = sp.coo_matrix((v, (i, j)), shape=shape)
    return D


def sparse2triplet(S):
    """"""
    return np.array(sp.find(S)).T


def negative_sampling(X):
    """"""
    pass


