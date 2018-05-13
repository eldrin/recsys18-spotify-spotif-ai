import os
from functools import partial
from scipy import sparse as sp
import numpy as np
import pandas as pd
import copy

import torch
from torch.autograd import Variable

from prefetch_generator import background
from util import sparse2triplet
# from evaluation import r_precision, NDCG
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from tqdm import trange, tqdm
import fire

import sys
sys.path.append(os.path.join(os.getcwd(), 'wmf'))
from wmf import factorize, cofactorize, cofactorize2
from wmf import log_surplus_confidence_matrix
from wmf import linear_surplus_confidence_matrix
from wmf import recompute_factors_bias
from wmf import iter_rows

sys.path.append('/home/ubuntu/workbench/RecsysChallengeTools/')
from metrics import ndcg, r_precision, playlist_extender_clicks
NDCG = partial(ndcg, k=500)

try:
    print(torch.cuda.current_device())
    floatX = torch.cuda.FloatTensor
except:
    floatX = torch.FloatTensor


def sigmoid(x):
    """"""
    return 1. / (1. + np.e ** -x)


def p_bar(X, u, i, j):
    """ posterior prob (true label) for u, i, j """
    if X[u, i] < X[u, j]:
        return 0
    elif X[u, i] > X[u, j]:
        return 1
    else:
        return .5


@background(max_prefetch=10)
def batch_prep_neg_sampling(X, x_trp, verbose=False, shuffle=True,
                            oversampling=10):
    """"""
    rnd_ix = np.random.choice(len(x_trp), len(x_trp), replace=False)

    if verbose:
        M = tqdm(rnd_ix, ncols=80)
    else:
        M = rnd_ix

    for m in M:

        # draw sample
        y = x_trp[m]

        # positive samples
        u, i = y[0], y[1]

        # sample negative sample
        for _ in xrange(oversampling):
            j_ = np.random.choice(X.shape[1])
            pos_i = set(sp.find(X[u])[1])
            while j_ in pos_i:
                j_ = np.random.choice(X.shape[1])
            yield u, i, j_


@background(max_prefetch=10)
def batch_prep_uniform_without_replace(X, x_trp, verbose=False, shuffle=True,
                                       oversampling=1):
    """"""
    M = X.shape[0]
    N = len(x_trp)  # number of records

    if verbose:
        it = tqdm(xrange(N), total=N, ncols=80)
    else:
        it = xrange(N)

    for n in it:
        # select user
        u = np.random.choice(M)
        i = sp.find(X[u])[1]
        if len(i) == 0:
            continue
        pos_i = set(i)
        i = np.random.choice(i)  # positive samples

        # sample negative sample
        for _ in xrange(oversampling):
            j_ = np.random.choice(X.shape[1])
            while j_ in pos_i:
                j_ = np.random.choice(X.shape[1])
            yield u, i, j_


@background(max_prefetch=10)
def batch_prep_fm(X, x_trp, batch_size=64, oversampling=20,
                  verbose=False, negative=True, shuffle=True):
    """"""
    M = X.shape[0]
    N = len(x_trp)  # number of records

    if verbose:
        it = tqdm(xrange(N), total=N, ncols=80)
    else:
        it = xrange(N)

    for n in it:
        trp = []
        y = []
        c = []  # confidence for weighting loss
        k = 0
        for _ in xrange(batch_size):
            # select user
            u = np.random.choice(M)
            i = sp.find(X[u])[1]
            if len(i) == 0:
                continue
            pos_i = set(i)
            i = np.random.choice(i)  # positive samples

            if negative:
                trp.extend([(k, u, 1), (k, i + M, 1)])
                y.append(1)
                c.append(X[u, i])
                k+=1

                for _ in xrange(oversampling):
                    # sample negative sample
                    j = np.random.choice(X.shape[1])
                    while j in pos_i:
                        j = np.random.choice(X.shape[1])

                    trp.extend([(k, u, 1), (k, j + M, 1)])
                    y.append(0)
                    c.append(X[u, j])
                    k+=1
            else:
                trp.extend([(k, u, 1), (k, i + M, 1)])
                y.append(1)
                c.append(X[u, i])
                k += 1

        # make batch with sparse matrix
        x = sp.coo_matrix(
            (map(lambda r: r[2], trp),
             (map(lambda r: r[0], trp),
              map(lambda r: r[1], trp))),
             shape=(k, sum(X.shape))
        ).toarray()
        yield x, y, c


def sparse_cosine_similarities(s_mat, axis=0):
    """"""
    if axis == 0:
        mat = s_mat.tocsc()
    elif axis == 1:
        mat = s_mat.tocsr()

    normed_mat = pp.normalize(mat, axis=axis)
    return normed_mat.T * normed_mat


class KNN:
    """"""
    def __init__(self, sim_fn=cosine_similarity, item_sim=True):
        """"""
        self.item_sim = item_sim
        self.sim_fn = sim_fn

    def fit(self, X, val=None):
        """"""
        if not self.item_sim:
            X = X.T

        X = X.tocsr()
        self.X = X
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(X)

    def predict_k(self, u, k=500):
        """"""
        dist, indx = self.nn.kneighbors(self.X[u], n_neighbors=k)
        return indx.ravel()


class BPRMF:
    """
    MF based on BPR loss.
    only covers SGD (m==1)
    """
    def __init__(self, n_components, init_factor=1e-1, alpha=1e-2, beta=0.0025,
                 n_epoch=2, optimizer=torch.optim.Adam,
                 dtype=floatX, verbose=False):
        """"""
        self.n_components_ = n_components
        self.n_epoch = n_epoch
        self.alpha = alpha  # learning rate
        self.beta = beta  # regularization weight
        self.init_factor = init_factor  # init weight
        self.dtype = dtype
        self.U = None  # u factors (torch variable / (n_u, n_r))
        self.V = None  # i factors (torch variable / (n_i, n_r))
        self.loss_curve = []
        self.optimizer = optimizer
        self.verbose = verbose

    def forward(self, u, i, j):
        """"""
        U, I, J = self.U[u], self.V[i], self.V[j]
        x_ui = (U * I).sum()
        x_uj = (U * J).sum()
        x_ = x_ui - x_uj

        L = -torch.log(torch.sigmoid(x_)).sum()
        L += self.beta * (torch.sum(U**2) + torch.sum(I**2) + torch.sum(J**2))
        return L

    def _preproc_data(self, X):
        """ unwrap sparse matrix X into triplets """
        Y = sparse2triplet(X)  # implement data processing
        # check zero columns and zero rows and make dummy factors

        return Y

    def predict_k(self, u, k=500):
        """"""
        r = self.U[u].matmul(self.V.t())
        _, ind = torch.sort(r)

        if self.dtype == torch.cuda.FloatTensor:
            ind_ = ind.cpu().data.numpy()
        else:
            ind_ = ind.data.numpy()

        return ind_[::-1][:k]

    def update(self, u, i, j):
        """"""
        U, I, J = self.U[u], self.V[i], self.V[j]
        x_ui = (U * I).sum()
        x_uj = (U * J).sum()
        x_ = x_ui - x_uj

        a = - 1. / (1. + torch.exp(x_))
        lr = self.alpha
        lda = self.beta

        # update user factor
        self.U[u].data.sub_(lr * (a * (I - J) + lda * (U**2).sum()).data)
        self.V[i].data.sub_(lr * (a * U + lda * (I**2).sum()).data)
        self.V[j].data.sub_(lr * (a * (-U) + lda * (J**2).sum()).data)

    def fit(self, X, val=None):
        """"""
        # quick convert
        X = X.tocsr()

        # setup relevant hyper params
        self.n_users = X.shape[0]
        self.n_items = X.shape[1]
        r = self.n_components_

        # init parameters
        self.U = Variable(
            torch.randn((self.n_users, r)).type(self.dtype) * self.init_factor,
            requires_grad=True
        )
        self.V = Variable(
            torch.randn((self.n_items, r)).type(self.dtype) * self.init_factor,
            requires_grad=True
        )

        # preprocess dataset
        Y = self._preproc_data(X)

        # # init optimizor
        # opt = self.optimizer([self.U, self.V], lr=self.alpha)

        # training loop
        if self.verbose:
            N = trange(self.n_epoch, ncols=80)
        else:
            N = xrange(self.n_epoch)

        try:
            for n in N:
                for u, i, j in batch_prep_neg_sampling(X, Y, self.verbose,
                                                       shuffle=True):
                    # # flush grad
                    # opt.zero_grad()

                    # forward pass
                    l = self.forward(u, i, j)

                    # save loss curve
                    l_ = l.data.cpu().numpy().item()
                    self.loss_curve.append(l_)

                    # # backward pass
                    # l.backward()

                    # # update
                    # opt.step()
                    self.update(u, i, j)

                    if self.verbose:
                        N.set_description(
                            '[loss : {:.4f}]'.format(l_)
                        )
        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')


class BPRMFcpu:
    """
    MF based on BPR loss.
    only covers SGD (m==1)
    experimental implementatio for speed comparison between numpy vs pytorch
    """
    def __init__(self, n_components, init_factor=1e-1, alpha=1e-2, beta=0.0025,
                 n_epoch=2, verbose=False, report_every=100):
        """"""
        self.n_components_ = n_components
        self.n_epoch = n_epoch
        self.alpha = alpha  # learning rate
        self.beta = beta  # regularization weight
        self.init_factor = init_factor  # init weight
        self.U = None  # u factors (torch variable / (n_u, n_r))
        self.V = None  # i factors (torch variable / (n_i, n_r))
        self.loss_curve = []
        self.acc_curve = []
        self.report_every = report_every
        self.verbose = verbose

    def predict_k(self, u, k=500):
        """"""
        r = self.U[u].dot(self.V.T) + self.b_i
        return np.argsort(r)[::-1][:k]

    def forward(self, u, i, j):
        """"""
        U, I, J = self.U[u], self.V[i], self.V[j]
        b_i = self.b_i[i]
        b_j = self.b_i[j]
        x_ui = np.inner(U, I)
        x_uj = np.inner(U, J)
        x_ = b_i - b_j + x_ui - x_uj

        L = np.log(1. / (1. + np.e ** (-x_)))
        L -= self.beta * (np.sum(U**2) + np.sum(I**2) + np.sum(J**2))
        L -= self.beta * .1 * np.sum(self.b_i[i]**2) + np.sum(self.b_i[j]**2)
        return L

    def update(self, u, i, j, beta=0.9, gamma=0.99, eps=1e-8):
        """"""
        U, I, J = self.U[u], self.V[i], self.V[j]
        b_i = self.b_i[i]
        b_j = self.b_i[j]
        x_ui = np.inner(U, I)
        x_uj = np.inner(U, J)
        x_ = b_i - b_j + x_ui - x_uj

        # get grad
        a = 1. / (1. + np.e ** (x_))
        di = a - self.beta * b_i
        dj = -a - self.beta * b_j
        dU = a * (I - J) - self.beta * U
        dI = a * U - self.beta * I
        dJ = a * (-U) - self.beta * J

        # # computing 1st and 2nd moment for each layer
        # self.U_m[u] = self.U_m[u] * beta + (1 - beta) * dU
        # self.V_m[i] = self.V_m[i] * beta + (1 - beta) * dI
        # self.V_m[j] = self.V_m[j] * beta + (1 - beta) * dJ
        # self.b_m[i] = self.b_m[i] * beta + (1 - beta) * di
        # self.b_m[j] = self.b_m[j] * beta + (1 - beta) * dj

        # self.U_v[u] = self.U_v[u] * gamma + (1 - gamma) * dU**2
        # self.V_v[i] = self.V_v[i] * gamma + (1 - gamma) * dI**2
        # self.V_v[j] = self.V_v[j] * gamma + (1 - gamma) * dJ**2
        # self.b_v[i] = self.b_v[i] * gamma + (1 - gamma) * di**2
        # self.b_v[j] = self.b_v[j] * gamma + (1 - gamma) * dj**2

        # # computing bias-corrected moment
        # U_m_corrected = self.U_m[u] / (1.-(beta ** self.k))
        # U_v_corrected = self.U_v[u] / (1.-(gamma ** self.k))

        # I_m_corrected = self.V_m[i] / (1.-(beta ** self.k))
        # I_v_corrected = self.V_v[i] / (1.-(gamma ** self.k))

        # J_m_corrected = self.V_m[j] / (1.-(beta ** self.k))
        # J_v_corrected = self.V_v[j] / (1.-(gamma ** self.k))

        # i_m_corrected = self.b_m[i] / (1.-(beta ** self.k))
        # i_v_corrected = self.b_v[i] / (1.-(gamma ** self.k))

        # j_m_corrected = self.b_m[j] / (1.-(beta ** self.k))
        # j_v_corrected = self.b_v[j] / (1.-(gamma ** self.k))

        # # update
        # U_update = U_m_corrected / (np.sqrt(U_v_corrected) + eps)
        # I_update = I_m_corrected / (np.sqrt(I_v_corrected) + eps)
        # J_update = J_m_corrected / (np.sqrt(J_v_corrected) + eps)
        # i_update = i_m_corrected / (np.sqrt(i_v_corrected) + eps)
        # j_update = j_m_corrected / (np.sqrt(j_v_corrected) + eps)

        # self.U[u] = self.U[u] + self.alpha * U_update
        # self.V[i] = self.V[i] + self.alpha * I_update
        # self.V[j] = self.V[j] + self.alpha * J_update
        # self.b_i[i] = self.b_i[i] + self.alpha * .1 * i_update
        # self.b_i[j] = self.b_i[j] + self.alpha * .1 * j_update

        # SGD
        self.b_i[i] += self.alpha * di
        self.b_i[j] += self.alpha * dj
        self.U[u] += self.alpha * dU
        self.V[i] += self.alpha * dI
        self.V[j] += self.alpha * dJ

    def _preproc_data(self, X):
        """ unwrap sparse matrix X into triplets """
        Y = sparse2triplet(X)  # implement data processing
        # check zero columns and zero rows and make dummy factors

        return Y

    def fit(self, X, val=None):
        """"""
        # quick convert
        X = X.tocsr()

        # setup relevant hyper params
        self.n_users = X.shape[0]
        self.n_items = X.shape[1]
        r = self.n_components_

        # init parameters
        self.b_u = None
        self.b_i = np.zeros(self.n_items)
        self.U = np.random.randn(self.n_users, r) * self.init_factor
        self.V = np.random.randn(self.n_items, r) * self.init_factor
        self.U_m, self.U_v = np.zeros(self.U.shape), np.zeros(self.U.shape)
        self.V_m, self.V_v = np.zeros(self.V.shape), np.zeros(self.V.shape)
        self.b_m, self.b_v = np.zeros(self.n_items), np.zeros(self.n_items)

        # preprocess dataset
        Y = self._preproc_data(X)

        # training loop
        if self.verbose:
            N = trange(self.n_epoch, ncols=80)
        else:
            N = xrange(self.n_epoch)

        self.k = 1
        try:
            for n in N:
                for u, i, j in batch_prep_uniform_without_replace(X, Y,
                                                                  self.verbose,
                                                                  shuffle=True):
                    if self.k % self.report_every == 0:
                        # forward pass
                        l = self.forward(u, i, j)

                        # save loss curve
                        self.loss_curve.append(l)

                        # diplay loss
                        if self.verbose:
                            N.set_description(
                                '[loss : {:.4f}]'.format(l)
                            )

                    if val is not None:
                        if self.k % (self.report_every * 1e+2) == 0:
                            rnd_u = np.random.choice(val.shape[0],
                                                     int(val.shape[0] * 0.05),
                                                     replace=False)
                            rprec = []
                            ndcg = []
                            for u_ in rnd_u:
                                true = sp.find(val[u_])[1]
                                pred = self.predict_k(u_, k=500)

                                rprec.append(r_precision(true, pred))
                                ndcg.append(NDCG(true, pred))
                            rprec = filter(lambda r: r is not None, rprec)
                            ndcg = filter(lambda r: r is not None, ndcg)
                            self.acc_curve.append(
                                (np.mean(rprec), np.mean(ndcg)))

                    # update
                    self.update(u, i, j)
                    self.k += 1

        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')


class LambdaBPRMF:
    """"""
    def __init__(self, n_components, alpha=0.001, beta=0.1, eps=1,
                 init_factor=1e-1, n_epoch=1, report_every=100,
                 prepare_psi_upto=100, verbose=True):
        """"""
        self.n_components_ = n_components
        self.n_epoch = n_epoch
        self.alpha = alpha  # learning rate
        self.beta = beta  # regularization weight
        self.eps = eps
        self.init_factor = init_factor  # init weight
        self.loss_curve = []
        self.acc_curve = []
        self.report_every = report_every
        self.verbose = verbose
        self.prepare_psi_upto = prepare_psi_upto

    def predict_k(self, u, k=500):
        """"""
        r = self.U[u].dot(self.V.T) + self.b_i
        return np.argsort(r)[::-1][:k]

    def predict(self, u, i):
        """"""
        r = self.U[u].dot(self.V[i]) + self.b_i[i]
        return r

    def p(self, u, i, j):
        """"""
        a_i = self.predict(u, i)
        a_j = self.predict(u, j)
        return sigmoid(a_i - a_j)

    def forward(self, u, i, j):
        """"""
        U, I, J = self.U[u], self.V[i], self.V[j]
        b_i = self.b_i[i]
        b_j = self.b_i[j]
        x_ui = np.inner(U, I)
        x_uj = np.inner(U, J)
        x_ = b_i - b_j + x_ui - x_uj

        L = np.log(1. / (1. + np.e ** (-x_)))
        L -= self.beta * (np.sum(U**2) + np.sum(I**2) + np.sum(J**2))
        L -= self.beta * .1 * np.sum(self.b_i[i]**2) + np.sum(self.b_i[j]**2)
        return L

    def update(self, u, i, j):
        """"""
        U, I, J = self.U[u], self.V[i], self.V[j]
        b_i = self.b_i[i]
        b_j = self.b_i[j]
        x_ui = np.inner(U, I)
        x_uj = np.inner(U, J)
        x_ = b_i - b_j + x_ui - x_uj

        # calc psi (lambda weight)
        T = copy.deepcopy(self.T)
        if int(T) in self.psi:
            psi = self.psi[int(T)]
        else:
            upto = int((self.n_items - 1.) / T) + 1
            psi = sum([1./(r+1.) for r in xrange(upto)]) / self.gamma_I

        # get grad (calc lambda)
        a = 1. / (1. + np.e ** (x_))
        a *= psi
        di = a - self.beta * b_i
        dj = -a - self.beta * b_j
        dU = a * (I - J) - self.beta * U
        dI = a * U - self.beta * I
        dJ = a * (-U) - self.beta * J

        # SGD
        self.b_i[i] += self.alpha * di
        self.b_i[j] += self.alpha * dj
        self.U[u] += self.alpha * dU
        self.V[i] += self.alpha * dI
        self.V[j] += self.alpha * dJ

    @background(max_prefetch=10)
    def batch_prep_lambda_scheme(self, X, x_trp):
        """"""
        M = X.shape[0]
        N = len(x_trp)  # number of records
        self.n_items = X.shape[1]
        self.n_users = M

        if self.verbose:
            it = tqdm(xrange(N), total=N, ncols=80)
        else:
            it = xrange(N)

        for n in it:
            # select user
            u = np.random.choice(M)
            i = sp.find(X[u])[1]
            if len(i) == 0:
                continue
            pos_i = set(i)
            i = np.random.choice(i)  # positive samples
            y_i = self.predict(u, i)

            # sample negative sample
            self.T = 1
            j = np.random.choice(self.n_items)
            y_j = self.predict(u, j)
            while (
                ((y_i - y_j > self.eps) or (p_bar(X, u, i, j) != 1)) and
                (self.T < (self.n_items - 1.))):

                j = np.random.choice(self.n_items)
                y_j = self.predict(u, j)
                self.T += 1

            if (y_i - y_j <= self.eps) and (p_bar(X, u, i, j) == 1):
                pass
            else:
                continue

            yield u, i, j

    def _preproc_data(self, X):
        """ unwrap sparse matrix X into triplets """
        Y = sparse2triplet(X)  # implement data processing
        return Y

    def fit(self, X, val=None):
        """"""
        # quick convert
        X = X.tocsr()

        # calc gamma_I
        self.gamma_I = sum([1./(r+1.) for r in xrange(X.shape[1])])

        # setup relevant hyper params
        self.n_users = X.shape[0]
        self.n_items = X.shape[1]
        r = self.n_components_

        # prepare psi per T
        self.psi = {}
        for t in xrange(1, self.prepare_psi_upto):
            upto = int((self.n_items - 1.) / t) + 1
            self.psi[t] = \
                    sum([1./(r_+1.) for r_ in xrange(upto)]) / self.gamma_I

        # init parameters
        self.b_u = None
        self.b_i = np.zeros((self.n_items,))

        self.U = np.random.randn(self.n_users, r) * self.init_factor
        self.V = np.random.randn(self.n_items, r) * self.init_factor
        self.U_m, self.U_v = np.zeros(self.U.shape), np.zeros(self.U.shape)
        self.V_m, self.V_v = np.zeros(self.V.shape), np.zeros(self.V.shape)
        self.b_m, self.b_v = np.zeros(self.n_items), np.zeros(self.n_items)

        # preprocess dataset
        Y = self._preproc_data(X)

        # training loop
        if self.verbose:
            N = trange(self.n_epoch, ncols=80)
        else:
            N = xrange(self.n_epoch)

        self.k = 1
        try:
            for n in N:
                for u, i, j in self.batch_prep_lambda_scheme(X, Y):
                    if self.k % self.report_every == 0:
                        # forward pass
                        l = self.forward(u, i, j)

                        # save loss curve
                        self.loss_curve.append(l)

                        # diplay loss
                        if self.verbose:
                            N.set_description(
                                '[loss : {:.4f}]'.format(l)
                            )

                    if val is not None:
                        if self.k % (self.report_every * 1e+2) == 0:
                            rnd_u = np.random.choice(val.shape[0],
                                                     int(val.shape[0] * 0.05),
                                                     replace=False)
                            rprec = []
                            ndcg = []
                            for u_ in rnd_u:
                                true = sp.find(val[u_])[1]
                                pred = self.predict_k(u_, k=500)

                                rprec.append(r_precision(true, pred))
                                ndcg.append(NDCG(true, pred))
                            rprec = filter(lambda r: r is not None, rprec)
                            ndcg = filter(lambda r: r is not None, ndcg)
                            self.acc_curve.append(
                                (np.mean(rprec), np.mean(ndcg)))

                    # update
                    self.update(u, i, j)
                    self.k += 1

        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')


class WRMF:
    """"""
    def __init__(self, n_components, init_factor=1e-1, beta_a=1, beta=1e-1,
                 gamma=1, epsilon=1, n_epoch=10, dtype='float32',
                 verbose=False, confidence_fn=log_surplus_confidence_matrix):
        """"""
        self.n_components_ = n_components
        self.n_epoch = n_epoch
        self.beta = beta  # regularization weight
        self.beta_a = beta_a
        self.gamma = gamma
        self.epsilon = epsilon
        self.init_factor = init_factor  # init weight
        self.confidence_fn = confidence_fn
        if confidence_fn == linear_surplus_confidence_matrix:
            self.confidence_fn = partial(confidence_fn, alpha=self.gamma)
        elif confidence_fn == log_surplus_confidence_matrix:
            self.confidence_fn = partial(
                confidence_fn, alpha=self.gamma, epsilon=self.epsilon)

        self.dtype = dtype
        self.U = None  # u factors (torch variable / (n_u, n_r))
        self.V = None  # i factors (torch variable / (n_i, n_r))
        self.loss_curve = []
        self.verbose = verbose

    def predict_k(self, u, k=500):
        """"""
        r = self.U[u].dot(self.V.T)
        return np.argsort(r)[::-1][:k]

    def fit(self, X, A=None, val=None):
        """
        X = interaction matrix
        A = attribute matrix for item (optional)
        """
        X = X.tocsr()
        S = self.confidence_fn(X)

        if A is None:
            UV = factorize(S, self.n_components_, lambda_reg=self.beta,
                           num_iterations=self.n_epoch, init_std=self.init_factor,
                           verbose=self.verbose, dtype=self.dtype)
            self.U_, self.V_ = UV
            self.U = self.U_
            self.V = self.V_
            self.W = None
        else:
            UVW = cofactorize(
                S, A, self.n_components_, lambda_a=self.beta_a,
                lambda_reg=self.beta, num_iterations=self.n_epoch,
                init_std=self.init_factor, verbose=self.verbose, dtype=self.dtype)
            self.U_, self.V_, self.W_ = UVW
            self.U = self.U_
            self.V = self.V_
            self.W = self.W_

        self.b_u = None
        self.b_i = None


class WRMF2:
    """"""
    def __init__(self, n_components, init_factor=1e-1, beta_a=1, beta_b=1, beta=1e-1,
                 gamma=1, epsilon=1, n_epoch=10, dtype='float32',
                 verbose=False, confidence_fn=log_surplus_confidence_matrix):
        """"""
        self.n_components_ = n_components
        self.n_epoch = n_epoch
        self.beta = beta  # regularization weight
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.gamma = gamma
        self.epsilon = epsilon
        self.init_factor = init_factor  # init weight
        self.confidence_fn = confidence_fn
        if confidence_fn == linear_surplus_confidence_matrix:
            self.confidence_fn = partial(confidence_fn, alpha=self.gamma)
        elif confidence_fn == log_surplus_confidence_matrix:
            self.confidence_fn = partial(
                confidence_fn, alpha=self.gamma, epsilon=self.epsilon)

        self.dtype = dtype
        self.U = None  # u factors (torch variable / (n_u, n_r))
        self.V = None  # i factors (torch variable / (n_i, n_r))
        self.loss_curve = []
        self.verbose = verbose

    def predict_k(self, u, k=500):
        """"""
        r = self.U[u].dot(self.V.T)
        return np.argsort(r)[::-1][:k]

    def fit(self, X, A, B, val=None):
        """
        X = interaction matrix
        A = attribute matrix for item (optional)
        """
        X = X.tocsr()
        S = self.confidence_fn(X)
        B = self.confidence_fn(B)

        UVW = cofactorize2(
            S, A, B, self.n_components_, lambda_a=self.beta_a, lambda_b=self.beta_b,
            lambda_reg=self.beta, num_iterations=self.n_epoch,
            init_std=self.init_factor, verbose=self.verbose, dtype=self.dtype)
        self.U_, self.V_, self.W_ = UVW
        self.U = self.U_
        self.V = self.V_
        self.W = self.W_

        self.b_u = None
        self.b_i = None



class FactorizationMachine:
    """"""
    def __init__(self, n_components, alpha=1e-1, beta=0, batch_size=16,
                 n_epoch=2, init_factor=1e-1, optimizer=torch.optim.Adam,
                 dtype=floatX, verbose=True):
        """"""
        self.n_components_ = n_components
        self.alpha = alpha
        self.beta = beta
        self.init_factor = init_factor
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.dtype = dtype
        self.optimizer = optimizer
        self.verbose = verbose
        self.loss_curve = []
        self.acc_curve = []

    def predict(self, x):
        """"""
        b = self.w0
        wx = x.matmul(self.w)
        vvxx = (x.matmul(self.v)**2 - (x**2).matmul(self.v**2)).sum(dim=-1) * .5
        return b + wx + vvxx

    def predict_k(self, u, k=500):
        """"""
        # for faster evaluation, pull data from gpu
        # and do simplified prediction
        if self.dtype == torch.cuda.FloatTensor:
            b = self.w0.data.cpu().numpy()
            w = self.w.data.cpu().numpy().ravel()
            v = self.v.data.cpu().numpy()
        else:
            b = self.w0.data.numpy()
            w = self.w.data.numpy().ravel()
            v = self.v.data.numpy()
        wx = w[u] + w[self.n_users:]
        vvxx = v[u][None].dot(v[self.n_users:].T) * .5
        out = b + wx + vvxx
        return np.argsort(out.ravel())[::-1][:k]

    def forward(self, x, y, c=None, criterion='mse'):
        """"""
        y_ = self.predict(x)
        s = 1. / (1. + torch.exp(-y_))
        if criterion == 'mse':
            L = (y - s)**2
            # L = (y - y_)**2
        elif criterion == 'ce':
            # L = torch.log(1. + torch.exp(-y * y_))
            L = -(y * torch.log(s) + (1 - y) * torch.log(1 - s))
            # L = -((1. - y) * y_ - torch.log(1. + torch.exp(-y_)))

        # apply surplus confidence function
        # (assuming they were already processed)
        if c is not None:
            L *= c

        # reduce the loss over the samples in the batch
        L = L.mean()

        # adding regularization terms
        L = L + self.beta * (
            ((self.w0**2).mean()) +
            ((self.w**2).mean()) +
            ((self.v**2).mean())
        )
        return L

    def _preproc_data(self, X):
        """ unwrap sparse matrix X into triplets """
        Y = sparse2triplet(X)  # implement data processing
        # check zero columns and zero rows and make dummy factors
        return Y

    def fit(self, X, val=None):
        """"""
        # quick convert
        X = X.tocsr()

        # setup relevant hyper params
        self.n_users = X.shape[0]
        self.n_items = X.shape[1]
        r = self.n_components_

        # init parameters
        self.w0 = Variable(
            torch.zeros((1,)).type(self.dtype),
            requires_grad=True
        )
        self.w = Variable(
            torch.zeros(
                (self.n_users + self.n_items, 1)).type(self.dtype),
            requires_grad=True
        )
        self.v = Variable(
            torch.randn(
                (self.n_users + self.n_items, r)
            ).type(self.dtype) * self.init_factor,
            requires_grad=True
        )

        # preprocess dataset
        Y = self._preproc_data(X)

        # init optimizor
        opt = self.optimizer([self.w0, self.w, self.v], lr=self.alpha)

        # training loop
        if self.verbose:
            N = trange(self.n_epoch, ncols=80)
        else:
            N = xrange(self.n_epoch)

        try:
            for n in N:
                for x, y, c in batch_prep_fm(X, Y, batch_size=self.batch_size,
                                             verbose=self.verbose):
                    # SGD
                    for xx, yy, cc in zip(x, y, c):
                        # cast to pytorch variable 
                        xx = Variable(torch.FloatTensor(xx).type(self.dtype))
                        yy = Variable(torch.FloatTensor([yy]).type(self.dtype))

                        # flush grad
                        opt.zero_grad()

                        # forward pass
                        # l = self.forward(xx, yy, c=1.+10.*cc)
                        l = self.forward(xx, yy)

                        # backward pass
                        l.backward()

                        # update
                        opt.step()

                    # # Mini-batch SGD
                    # # cast to pytorch variable
                    # x = Variable(torch.FloatTensor(x).type(self.dtype))
                    # y = Variable(torch.FloatTensor(y).type(self.dtype))

                    # # flush grad
                    # opt.zero_grad()

                    # # forward
                    # l = self.forward(x, y)

                    # # backward
                    # l.backward()

                    # # update
                    # opt.step()

                    if self.dtype == torch.cuda.FloatTensor:
                        l_ = l.cpu().data.numpy()
                    else:
                        l_ = l.data.numpy()

                    # save loss curve
                    self.loss_curve.append(l_)

                    if self.verbose:
                        N.set_description(
                            '[loss : {:.4f}]'.format(l_.item())
                        )
        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')


def naive_boosting_Uc(S, A, Uc, U, V, W, lambda_reg, dtype='float32'):
    """"""
    S, A = S.tocsr(), A.tocsr()

    m = S.shape[0]
    f = U.shape[1]

    Vw = A.T.dot(W)
    VTVw = np.dot(V.T, Vw)

    VwTVw = np.dot(Vw.T, Vw)
    VwTVwpI = VwTVw + lambda_reg * np.eye(f)

    Uc_new = np.zeros((m, f))

    for k, s_u, i_u in iter_rows(S):
        U_u = U[k]
        V_u = V[i_u]
        Vw_u = Vw[i_u]
        VTSVw = np.dot(V_u.T, (Vw_u * s_u.reshape(-1, 1)))
        A = np.dot(s_u + 1, Vw_u) - np.dot(U_u, VTVw + VTSVw)
        VwTSVw = np.dot(Vw_u.T, (Vw_u * s_u.reshape(-1, 1)))
        B = VwTVw + VwTVwpI

        Uc_new[k] = np.linalg.solve(B.T, A.T).T

    return Uc_new


def boosted_pred_k(u, model, Uc, Vw, k=500):
    """"""
    r = model.U[u].dot(model.V.T) + Uc[u].dot(Vw.T)
    return np.argsort(r)[::-1][:k]


def main(train_data_fn, test_data_fn, r=2, attr_fn=None, attr_fn2=None):
    """"""
    from util import read_data
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('Loading data...')
    cutoff = 500
    # d = read_data(data_fn, delimiter='\t')
    d = read_data(train_data_fn).tocsr()
    dt = read_data(test_data_fn, shape=d.shape).tocsr()

    if attr_fn is not None:
        a = read_data(attr_fn)
        a_n = a.sum(axis=1)
    if attr_fn2 is not None:
        b = read_data(attr_fn2, shape=(a.shape[0], a.shape[0]))


    print('Fit model (r={:d})!'.format(r))
    # fit
    # model = KNN()
    # model = BPRMF(r, verbose=True)
    # model = WRMF(r, beta_a=1, beta=1, gamma=100, epsilon=1e-1, n_epoch=10, verbose=True)
    model = WRMF2(r, beta_a=1, beta_b=1, beta=1, gamma=100, epsilon=1e-1, n_epoch=20, verbose=True)
    # model = BPRMFcpu(r, alpha=1e-1, beta=1e-5, n_epoch=2, verbose=True)
    # model = LambdaBPRMF(10, alpha=0.005, beta=5, n_epoch=2, verbose=True)
    # model = FactorizationMachine(10, alpha=1e-2, beta=0.01, batch_size=4, verbose=True)

    boosted = False
    if attr_fn is not None and attr_fn2 is None:
        model.fit(d, a, val=None)
    elif attr_fn is not None and attr_fn2 is not None:
        model.fit(d, a, b, val=None)

        print('Naive boosting...')
        Uc = np.random.randn(*model.U.shape) * 1e-2
        for i in range(5):
            print('[boosting] {:d}th iter...'.format(i))
            Uc = naive_boosting_Uc(
                d, a, Uc, model.U, model.V, model.W, model.beta)
        Vw = a.T.dot(model.W)
        boosted = True

    else:
        print('No Attribute Information!')
        model.fit(d, val=None)

    print('Evaluate!')
    # predict
    # for efficient, sample 5% of the user to approximate the performance
    # rnd_u = np.random.choice(d.shape[0], int(d.shape[0] * 0.05), replace=False)
    trg_u = np.where(dt.sum(axis=1) > 0)[0]
    rprec = []
    ndcg_ = []
    clicks = []
    for u in tqdm(trg_u, total=len(trg_u), ncols=80):
        true = sp.find(dt[u])[1]
        if len(true) == 0:
            continue
        pred = model.predict_k(u, k=cutoff * 2)
        # exclude training data
        true_t = set(sp.find(d[u])[1])
        pred = filter(lambda x: x not in true_t, pred)[:cutoff]

        rprec.append(r_precision(true, pred))
        ndcg_.append(NDCG(true, pred))
        clicks.append(playlist_extender_clicks(true, pred))

    rprec = filter(lambda r: r is not None, rprec)
    ndcg_ = filter(lambda r: r is not None, ndcg_)
    clicks = filter(lambda r: r is not None, clicks)

    print('R Precision: {:.4f}'.format(np.mean(rprec)))
    print('NDCG: {:.4f}'.format(np.mean(ndcg_)))
    print('Playlist Extender Clicks: {:.4f}'.format(np.mean(clicks)))

    if boosted:
        print('Boosted!!')
        rprec = []
        ndcg_ = []
        clicks = []
        for u in tqdm(trg_u, total=len(trg_u), ncols=80):
            true = sp.find(dt[u])[1]
            if len(true) == 0:
                continue
            pred = boosted_pred_k(u, model, Uc, Vw, k=cutoff * 2)
            # exclude training data
            true_t = set(sp.find(d[u])[1])
            pred = filter(lambda x: x not in true_t, pred)[:cutoff]

            rprec.append(r_precision(true, pred))
            ndcg_.append(NDCG(true, pred))
            clicks.append(playlist_extender_clicks(true, pred))

        rprec = filter(lambda r: r is not None, rprec)
        ndcg_ = filter(lambda r: r is not None, ndcg_)
        clicks = filter(lambda r: r is not None, clicks)

        print('R Precision: {:.4f}'.format(np.mean(rprec)))
        print('NDCG: {:.4f}'.format(np.mean(ndcg_)))
        print('Playlist Extender Clicks: {:.4f}'.format(np.mean(clicks)))

    print('Save models!')
    np.save('./data/bpr_U.npy', model.U)
    np.save('./data/bpr_V.npy', model.V)
    if hasattr(model, 'W') and model.W is not None:
        np.save('./data/bpr_W.npy', model.W)

    if boosted:
        np.save('./data/bpr_Uc.npy', Uc)
        np.save('./data/bpr_Vw.npy', Vw)

    if model.b_i is not None:
        np.save('./data/bpr_b_i.npy', model.b_i)
    if model.b_u is not None:
        np.save('./data/bpr_b_u.npy', model.b_u)

    fig, ax = plt.subplots(1, 1)
    ax.plot(model.loss_curve, 'x')
    fig.savefig('./data/loss.png')

    if hasattr(model, 'acc_curve'):
        fig, ax = plt.subplots(1, 1)
        ax.plot(map(lambda x: x[0], model.acc_curve), 'x')
        fig.savefig('./data/rprec.png')

        fig, ax = plt.subplots(1, 1)
        ax.plot(map(lambda x: x[1], model.acc_curve), 'x')
        fig.savefig('./data/ndcg.png')


def test_UV(all_data_fn, test_data_fn, fn_U, fn_V):
    """"""
    from util import read_data
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('Loading data...')
    cutoff = 500

    d = read_data(all_data_fn)
    dt = read_data(test_data_fn, delimiter='\t', shape=d.shape).tocsr()

    U = np.load(fn_U)
    V = np.load(fn_V)
    model = WRMF(U.shape[1])
    model.U = U
    model.V = V

    print('Evaluate!')
    # predict
    # for efficient, sample 5% of the user to approximate the performance
    rnd_u = np.random.choice(d.shape[0], int(d.shape[0] * 0.05), replace=False)
    rprec = []
    ndcg = []
    for u in tqdm(rnd_u, total=len(rnd_u), ncols=80):
        true = sp.find(dt[u])[1]
        pred = model.predict_k(u, k=cutoff)

        rprec.append(r_precision(true, pred))
        ndcg.append(NDCG(true, pred))
    rprec = filter(lambda r: r is not None, rprec)
    ndcg = filter(lambda r: r is not None, ndcg)

    print('R Precision: {:.4f}'.format(np.mean(rprec)))
    print('NDCG: {:.4f}'.format(np.mean(ndcg)))


def test_libfm(data_fn, r=2, attr_fn=None):
    """"""
    from util import read_data
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print('Loading data...')
    cutoff = 500
    # d = read_data(data_fn, delimiter='\t')
    d = read_data(data_fn)
    if attr_fn is not None:
        a = read_data(attr_fn).tocsc()
        A = {}
        ind = np.array(sp.find(a)).T
        for i in xrange(a.shape[1]):
            A[i] = ind[ind[:, 1] == i][:, 0]
    else:
        a = None

    i, j, v = sp.find(d)
    rnd_idx = np.random.choice(len(i), len(i), replace=False)
    bound = int(len(i) * 0.7)
    rnd_idx_trn = rnd_idx[:bound]
    rnd_idx_val = rnd_idx[bound:]
    d = sp.coo_matrix((v[rnd_idx_trn], (i[rnd_idx_trn], j[rnd_idx_trn])),
                      shape=d.shape).tocsr()
    dt = sp.coo_matrix((v[rnd_idx_val], (i[rnd_idx_val], j[rnd_idx_val])),
                       shape=d.shape).tocsr()

    # make fm dataset (X: csc_matrix / y: mat (n_compares, 2))
    # (including negative sampling)
    print('Preparing Training Dataset')
    n_users = max(i) + 1
    n_items = max(j) + 1
    n_attrs = a.shape[0] if a is not None else None
    trp_trn = []
    R = []
    k = 0
    for u in tqdm(xrange(n_users), total=n_users, ncols=80):
        pos_i = set(sp.find(d[u])[1])
        if len(pos_i) == 0:
            continue
        for i_ in pos_i:
            if a is not None:
                for aa_ in A[i_]:
                    trp_trn.append((k, aa_ + n_users + n_items, 1))
            trp_trn.extend([(k, u, 1), (k, i_ + n_users, 1)])
            # R.append(d[u, i_])
            R.append(1)
            k += 1

            # sample negative sample
            for _ in xrange(20):  # oversampling
                j_ = np.random.choice(d.shape[1])
                while j_ in pos_i:
                    j_ = np.random.choice(d.shape[1])

                if a is not None:
                    for aa_ in A[j_]:
                        trp_trn.append((k, aa_ + n_users + n_items, 1))
                trp_trn.extend([(k, u, 1), (k, j_ + n_users, 1)])
                # R.append(d[u, j_])
                R.append(0)
                k += 1

    # wrap it into sparse matrix
    if a is not None:
        shape_ = (k, n_users + n_items + n_attrs)
    else:
        shape_ = (k, n_users + n_items)

    X = sp.coo_matrix(
        (map(lambda x: x[2], trp_trn),
         (map(lambda x: x[0], trp_trn),
          map(lambda x: x[1], trp_trn))),
        shape=shape_
    ).tocsc()
    print('Train data : ({:d}, {:d})'.format(*X.shape))

    # prepare test data
    print('Preparing Testing Dataset')
    rnd_u = np.random.choice(d.shape[0], int(d.shape[0] * 0.01), replace=False)
    trp = []
    yt = []
    k = 0
    for u in tqdm(rnd_u, total=len(rnd_u), ncols=80):
        for i in xrange(n_items):
            if a is not None:
                for aa_ in A[i_]:
                    trp.append((k, aa_ + n_users + n_items, 1))
            trp.extend([(k, u, 1), (k, i + n_users, 1)])
            # yt.append(d[u, i])
            yt.append(d[u, i] > 0)
            k += 1

    if a is not None:
        shape_ = (k, n_users + n_items + n_attrs)
    else:
        shape_ = (k, n_users + n_items)

    Xt = sp.coo_matrix(
        (map(lambda x: x[2], trp),
         (map(lambda x: x[0], trp),
          map(lambda x: x[1], trp))),
        shape=shape_
    )
    print('Test data : ({:d}, {:d})'.format(*Xt.shape))

    # instantiate models
    print('Training Model!')
    from pywFM import FM
    fm = FM(task='regression', num_iter=100, k2=r, verbose=False, rlog=False)
    model = fm.run(X, R, Xt, yt)

    print('Evaluate!')
    rprec = []
    ndcg = []
    for u, uu in zip(rnd_u, xrange(0, n_items * len(rnd_u), n_items)):
        true = sp.find(dt[u])[1]
        pred = np.argsort(model.predictions[slice(uu, uu + n_items)])
        pred = pred[::-1][:cutoff]

        rprec.append(r_precision(true, pred))
        ndcg.append(NDCG(true, pred))
    rprec = filter(lambda r: r is not None, rprec)
    ndcg = filter(lambda r: r is not None, ndcg)

    print('R Precision: {:.4f}'.format(np.mean(rprec)))
    print('NDCG: {:.4f}'.format(np.mean(ndcg)))


if __name__ == "__main__":
    fire.Fire(main)
    # fire.Fire(test_libfm)
    # fire.Fire(test_UV)
