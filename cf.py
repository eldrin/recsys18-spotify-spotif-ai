import os
from functools import partial
from scipy import sparse as sp
import numpy as np

import torch
from torch.autograd import Variable

from prefetch_generator import background
from util import sparse2triplet
from evaluation import r_precision, NDCG

from tqdm import trange, tqdm
import fire

import sys
sys.path.append(os.path.join(os.getcwd(), 'wmf'))
from wmf import factorize
from wmf import log_surplus_confidence_matrix
from wmf import linear_surplus_confidence_matrix
from wmf import recompute_factors_bias

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
                                       oversampling=10):
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

    def fit(self, X):
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
                    self.loss_curve.append(l.data)

                    # # backward pass
                    # l.backward()

                    # # update
                    # opt.step()
                    self.update(u, i, j)

                    if self.verbose:
                        N.set_description(
                            '[loss : {:.4f}]'.format(l.data.numpy().item())
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
        if int(self.T) in self.psi:
            psi = self.psi[int(self.T)]
        else:
            upto = int((self.n_items - 1.) / self.T) + 1
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
    def __init__(self, n_components, init_factor=1e-1, beta=1e-1,
                 gamma=1, epsilon=1, n_epoch=10, dtype='float32',
                 verbose=False, confidence_fn=log_surplus_confidence_matrix):
        """"""
        self.n_components_ = n_components
        self.n_epoch = n_epoch
        self.beta = beta  # regularization weight
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

    def fit(self, X, val=None):
        """"""
        X = X.tocsr()
        S = self.confidence_fn(X)
        UV = factorize(S, self.n_components_, lambda_reg=self.beta,
                       num_iterations=self.n_epoch, init_std=self.init_factor,
                       verbose=self.verbose, dtype=self.dtype)
        self.U_, self.V_ = UV
        self.U = self.U_
        self.V = self.V_
        self.b_u = None
        self.b_i = None


def main(data_fn):
    """"""
    from util import read_data
    import matplotlib.pyplot as plt

    print('Loading data...')
    cutoff = 500
    # d = read_data(data_fn, delimiter='\t')
    d = read_data(data_fn)
    i, j, v = sp.find(d)
    rnd_idx = np.random.choice(len(i), len(i), replace=False)
    bound = int(len(i) * 0.7)
    rnd_idx_trn = rnd_idx[:bound]
    rnd_idx_val = rnd_idx[bound:]
    d = sp.coo_matrix((v[rnd_idx_trn], (i[rnd_idx_trn], j[rnd_idx_trn])),
                      shape=d.shape)
    dt = sp.coo_matrix((v[rnd_idx_val], (i[rnd_idx_val], j[rnd_idx_val])),
                       shape=d.shape).tocsr()

    print('Fit model!')
    # fit
    # model = BPRMF(10, alpha=0.003, beta=0.001, verbose=True)
    # model = WRMF(10, beta=1e-1, verbose=True)
    # model = BPRMFcpu(10, alpha=0.004, beta=10, n_epoch=2, verbose=True)
    model = LambdaBPRMF(10, alpha=0.1, beta=0.001, n_epoch=2, verbose=True)
    model.fit(d, dt)

    print('Evaluate!')
    # predict
    # for efficient, sample 5% of the user to approximate the performance
    rnd_u = np.random.choice(d.shape[0], int(d.shape[0] * 0.1), replace=False)
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

    print('Save models!')
    np.save('./data/bpr_U.npy', model.U)
    np.save('./data/bpr_V.npy', model.V)

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


if __name__ == "__main__":
    fire.Fire(main)
