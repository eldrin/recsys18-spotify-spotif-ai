import os
from scipy import sparse as sp
import numpy as np

import torch
from torch.autograd import Variable

from prefetch_generator import background
from util import sparse2triplet

from tqdm import trange, tqdm
import fire

import sys
sys.path.append(os.path.join(os.getcwd(), 'wmf'))
from wmf import factorize, log_surplus_confidence_matrix

try:
    print(torch.cuda.current_device())
    floatX = torch.cuda.FloatTensor
except:
    floatX = torch.FloatTensor


def sigmoid(x):
    """"""
    return 1. / (1. + np.exp(-x))


@background(max_prefetch=10)
def batch_prep_neg_sampling(X, x_trp, verbose=False, shuffle=True):
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
        j = np.random.choice(X.shape[1])
        pos_i = set(sp.find(X[u])[1])
        while j in pos_i:
            j = np.random.choice(X.shape[1])

        yield u, i, j

@background(max_prefetch=10)
def batch_prep_uniform_without_replace(X, x_trp, verbose=False, shuffle=True):
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
        i = np.random.choice(sp.find(X[u])[1]) # positive samples

        # sample negative sample
        j = np.random.choice(X.shape[1])
        pos_i = set(sp.find(X[u])[1])
        while j in pos_i:
            j = np.random.choice(X.shape[1])

        yield u, i, j

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

    def predict(self, u, k=500):
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
                    self.update(u, i ,j)

                    if self.verbose:
                        N.set_description(
                            '[loss : {:.4f}]'.format(l.data.numpy().item())
                        )
        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')


class BPRMF_numpy:
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
        self.report_every = report_every
        self.verbose = verbose

    def predict(self, u, k=500):
        """"""
        r = self.U[u].dot(self.V.T)
        return np.argsort(r)[::-1][:k]

    def forward(self, u, i, j):
        """"""
        U, I, J = self.U[u], self.V[i], self.V[j]
        x_ui = np.inner(U, I)
        x_uj = np.inner(U, J)
        x_ = x_ui - x_uj

        L = -np.log(sigmoid(x_))
        L += self.beta * (np.sum(U**2) + np.sum(I**2) + np.sum(J**2))
        return L

    def update(self, u, i, j, beta=0.9, gamma=0.99, eps=1e-8):
        """"""
        U, I, J = self.U[u], self.V[i], self.V[j]
        x_ui = np.inner(U, I)
        x_uj = np.inner(U, J)
        x_ = x_ui - x_uj

        # get grad
        a = 1. / (1. + np.e ** (x_))
        dU = a * (I - J) - self.beta * U
        dI = a * U - self.beta * I
        dJ = a * (-U) - self.beta * J

        # # computing 1st and 2nd moment for each layer
        # self.U_m[u] = self.U_m[u] * beta + (1 - beta) * dU
        # self.V_m[i] = self.V_m[i] * beta + (1 - beta) * dI
        # self.V_m[j] = self.V_m[j] * beta + (1 - beta) * dJ

        # self.U_v[u] = self.U_v[u] * gamma + (1 - gamma) * dU**2
        # self.V_v[i] = self.V_v[i] * gamma + (1 - gamma) * dI**2
        # self.V_v[j] = self.V_v[j] * gamma + (1 - gamma) * dJ**2

        # # computing bias-corrected moment
        # U_m_corrected = self.U_m[u] / (1.-(beta ** self.k))
        # U_v_corrected = self.U_v[u] / (1.-(gamma ** self.k))

        # I_m_corrected = self.V_m[i] / (1.-(beta ** self.k))
        # I_v_corrected = self.V_v[i] / (1.-(gamma ** self.k))

        # J_m_corrected = self.V_m[j] / (1.-(beta ** self.k))
        # J_v_corrected = self.V_v[j] / (1.-(gamma ** self.k))

        # # update
        # U_update = U_m_corrected / (np.sqrt(U_v_corrected) + eps)
        # I_update = I_m_corrected / (np.sqrt(I_v_corrected) + eps)
        # J_update = J_m_corrected / (np.sqrt(J_v_corrected) + eps)

        # self.U[u] = self.U[u] - self.alpha * U_update
        # self.V[i] = self.V[i] - self.alpha * I_update
        # self.V[j] = self.V[j] - self.alpha * J_update

        # SGD
        self.U[u] = self.U[u].copy() + self.alpha * dU
        self.V[i] = self.V[i].copy() + self.alpha * dI
        self.V[j] = self.V[j].copy() + self.alpha * dJ

    def _preproc_data(self, X):
        """ unwrap sparse matrix X into triplets """
        Y = sparse2triplet(X)  # implement data processing
        # check zero columns and zero rows and make dummy factors

        return Y

    def fit(self, X):
        """"""
        # quick convert
        X = X.tocsr()

        # setup relevant hyper params
        self.n_users = X.shape[0]
        self.n_items = X.shape[1]
        r = self.n_components_

        # init parameters
        self.U = np.random.randn(self.n_users, r) * self.init_factor
        self.V = np.random.randn(self.n_items, r) * self.init_factor
        self.U_m, self.U_v = np.zeros(self.U.shape), np.zeros(self.U.shape)
        self.V_m, self.V_v = np.zeros(self.V.shape), np.zeros(self.V.shape)

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
                # for u, i, j in batch_prep_neg_sampling(X, Y, self.verbose,
                #                                        shuffle=True):
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

                    # update
                    self.update(u, i, j)
                    self.k += 1

        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')


class WRMF:
    """"""
    def __init__(self, n_components, init_factor=1e-3, beta=1e-1,
                 gamma=1, epsilon=1, n_epoch=5, dtype='float32',
                 verbose=False):
        """"""
        self.n_components_ = n_components
        self.n_epoch = n_epoch
        self.beta = beta  # regularization weight
        self.gamma = gamma
        self.epsilon = epsilon
        self.init_factor = init_factor  # init weight
        self.dtype = dtype
        self.U = None  # u factors (torch variable / (n_u, n_r))
        self.V = None  # i factors (torch variable / (n_i, n_r))
        self.loss_curve = []
        self.verbose = verbose

    def predict(self, u, k=500):
        """"""
        r = self.U[u].dot(self.V.T)
        return np.argsort(r)[::-1][:k]

    def fit(self, X):
        """"""
        X = X.tocsr()
        S = log_surplus_confidence_matrix(X, self.gamma, self.epsilon)
        UV = factorize(S, self.n_components_, lambda_reg=self.beta,
                       num_iterations=self.n_epoch, init_std=self.init_factor,
                       verbose=self.verbose, dtype=self.dtype)
        self.U, self.V = UV


def main(data_fn):
    """"""
    from util import read_data
    from evaluation import r_precision, NDCG
    import matplotlib.pyplot as plt

    print('Loading data...')
    cutoff = 500
    # d = read_data(data_fn, delimiter='\t')
    d = read_data(data_fn)
    i, j, v = sp.find(d)
    rnd_idx = np.random.choice(len(i), len(i), replace=False)
    bound = int(len(i) * 0.8)
    rnd_idx_trn = rnd_idx[:bound]
    rnd_idx_val = rnd_idx[bound:]
    d = sp.coo_matrix((v[rnd_idx_trn], (i[rnd_idx_trn], j[rnd_idx_trn])),
                      shape=d.shape)
    dt = sp.coo_matrix((v[rnd_idx_val], (i[rnd_idx_val], j[rnd_idx_val])),
                       shape=d.shape).tocsr()

    print('Fit model!')
    # fit
    model = BPRMF(10, alpha=0.003, beta=0.001, init_factor=1e-2, verbose=True)
    # model = WRMF(10, verbose=True)
    # model = BPRMF_numpy(10, alpha=0.003, beta=0.001, init_factor=1e-2, verbose=True)
    model.fit(d)

    print('Evaluate!')
    # predict
    # for efficient, sample 5% of the user to approximate the performance
    rnd_u = np.random.choice(d.shape[0], int(d.shape[0] * 0.05), replace=False)
    rprec = []
    ndcg = []
    for u in tqdm(rnd_u, total=len(rnd_u), ncols=80):
        true = sp.find(dt[u])[1]
        pred = model.predict(u, k=cutoff)

        rprec.append(r_precision(true, pred))
        ndcg.append(NDCG(true, pred))
    rprec = filter(lambda r: r is not None, rprec)
    ndcg = filter(lambda r: r is not None, ndcg)

    print('R Precision: {:.4f}'.format(np.mean(rprec)))
    print('NDCG: {:.4f}'.format(np.mean(ndcg)))

    fig, ax = plt.subplots(1, 1)
    ax.plot(model.loss_curve)
    fig.savefig('./data/loss.png')


if __name__ == "__main__":
    fire.Fire(main)
