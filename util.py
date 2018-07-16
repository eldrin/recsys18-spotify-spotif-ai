from itertools import chain
from functools import partial
from scipy import sparse as sp
import pandas as pd
import numpy as np
import torch


TORCH_DTYPE_TENSOR = {
    'float': torch.FloatTensor,
    'int': torch.LongTensor
}


def flatten(nested_list):
    """"""
    return list(chain.from_iterable(nested_list))


def numpy2torchvar(ndarray, dtype='float', gpu=True):
    tensor = TORCH_DTYPE_TENSOR[dtype](ndarray)
    if gpu:
        return torch.autograd.Variable(tensor.cuda())
    else:
        return torch.autograd.Variable(tensor)


def read_data(fn, delimiter=',', f_cast=int, shape=None):
    """"""
    ll = pd.read_csv(fn, sep=delimiter, header=None, index_col=None)
    # build sparse matrix excluding zero entries
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
    D.eliminate_zeros()
    return D, ll


def read_hash(fn):
    """"""
    with open(fn) as f:
        l = [ll.replace('\n','').split('\t') for ll in f.readlines()]
    return pd.DataFrame(l)


def sparse2triplet(S):
    """"""
    return np.array(sp.find(S)).T


def beta_sigmoid(x, beta=2):
    """"""
    explogit = np.exp(beta * x)
    return 2 / (1. + explogit)


def blend_beta_sigmoid(y, U1, U2, b):
    """"""
    _b_sigm = partial(beta_sigmoid, beta=b)
    g = y.groupby('playlist')['value'].sum().apply(_b_sigm).values[:, None]
    return ((1-g) * U1 + g * U2).astype(np.float32)


def sigmoid(x):
    """"""
    return 1./(1. + np.exp(-x))


class BaseMF:
    """"""
    def __init__(self, P, Q, importance=1, logistic=False, name='MF', **kwargs):
        """"""
        # fill the 0 if there are
        zero_ix = P.sum(axis=1) == 0
        P[zero_ix] = np.random.randn(*P[zero_ix].shape) * 0.01

        self.P = P
        self.Q = Q
        self.logistic = logistic
        self.a = importance
        self.name = name

    def predict_score(self, u):
        """"""
        score = self.P[u].dot(self.Q.T)
        return sigmoid(score) if self.logistic else score


class MultiMF:
    """"""
    def __init__(self, *mfmodels):
        """"""
        self.models = list(mfmodels)

    def predict_k(self, pids, k=500):
        """"""
        # get scores
        scores = -np.sum(
            [mf.a * mf.predict_score(pids) for mf in self.models], axis=0
        )
        if (isinstance(pids, (int, float)) or
            (isinstance(pids, (list, tuple)) and len(pids) == 1)):
            scores = scores[None, :]

        ix = np.argpartition(scores, k, axis=1)[:, :k]
        pred_raw_ix = scores[np.arange(ix.shape[0])[:,None], ix].argsort(1)
        pred_raw_batch = ix[np.arange(ix.shape[0])[:,None], pred_raw_ix]

        return pred_raw_batch


def load_checkpoint(model, checkpoint_fn):
    """"""
    checkpoint = torch.load(checkoint_fn)
    model.eval()
    model.load_state_dict(checkpoint['sate_dict'])


def load_libfm_model(model_fn, verbose=False):
    """"""
    # preprocessing
    with open(model_fn) as f:
        i = 0
        bounds = []
        for line in f:
            if line[0] == '#':
                bounds.append(i)
            i+=1

    # main processing loop
    w0 = 0
    w1 = []
    w2 = []
    with open(model_fn) as f:
        i = 0
        if verbose:
            iterator = tqdm(f, total=bounds[2]*2 - 3, ncols=80)
        else:
            iterator = f

        for line in iterator:
            if i == bounds[0] + 1:
                w0 = np.float32(line.replace('\n', ''))
            elif i > bounds[1] and i < bounds[2]:
                w1.append(np.float32(line.replace('\n', '')))
            elif i > bounds[2]:
                w2.append([np.float32(l)
                           for l
                           in line.replace('\n', '').split()])
            i+=1

    # post-processing: convert lists into numpy arrays
    w1 = np.array(w1)
    w2 = np.array(w2)

    return w0, w1, w2


def libfm2uv(libfm_model_fn, n_users, n_items):
    """"""
    w0, w1, w2 = load_libfm_model(libfm_model_fn)

    # convert them into two matrices
    bias_u = w1[:n_users]
    bias_i = w1[n_users:]
    factors_u = w2[:n_users]
    factors_i = w2[n_users:]

    U = np.c_[factors_u, bias_u, np.ones(bias_u.shape)]
    V = np.c_[factors_i, np.ones(bias_i.shape), bias_i]

    return U.astype(np.float32), V.astype(np.float32)


def convert_libfm_model_n_save(libfm_model_fn, n_users, n_items, out_fn_U, out_fn_V):
    """"""
    U, V = libfm2uv(libfm_model_fn, n_users, n_items)
    np.save(out_fn_U, U)
    np.save(out_fn_V, V)
