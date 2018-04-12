from scipy import sparse as sp
import pandas as pd
import numpy as np


def read_data(fn, delimiter=',', f_cast=int, shape=None):
    """"""
    # with open(fn) as f:
    #     ll = [
    #         map(f_cast, l_.replace('\n','').split(delimiter))
    #         for l_ in f.readlines()
    #     ]
    # offset_i = min(map(lambda x: x[0], ll))
    # offset_j = min(map(lambda x: x[1], ll))

    ll = pd.read_csv(fn, sep=delimiter, header=None)
    offset_i = ll[0].min()
    offset_j = ll[1].min()

    # i = map(lambda x: x[0] - offset_i, ll)
    # j = map(lambda x: x[1] - offset_j, ll)
    i = ll[0] - offset_i
    j = ll[1] - offset_j
    if len(ll.columns) > 2:
        # v = map(lambda x: x[2], ll)
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


