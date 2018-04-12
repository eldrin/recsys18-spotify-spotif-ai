from scipy import sparse as sp
import torch
from torch.autograd import Variable
from util import sparse2triplet
import numpy as np
from tqdm import trange, tqdm

from prefetch_generator import background


@background(max_prefetch=3)
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


class BPRMF:
    """
    MF based on BPR loss.
    only covers SGD (m==1)
    """
    def __init__(self, n_components, init_factor=1e-1, alpha=1e-6, beta=1e+3,
                 n_epoch=2, optimizer=torch.optim.Adam, verbose=False):
        """"""
        self.n_components_ = n_components
        self.n_epoch = n_epoch
        self.alpha = alpha  # learning rate
        self.beta = beta  # regularization weight
        self.init_factor = init_factor  # init weight
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
        return np.argsort(self.U[u].matmul(self.V.t()).data.numpy())[:k][::-1]


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
            torch.randn((self.n_users, r)) * self.init_factor,
            requires_grad=True
        )
        self.V = Variable(
            torch.randn((self.n_items, r)) * self.init_factor,
            requires_grad=True
        )

        # preprocess dataset 
        Y = self._preproc_data(X)

        # init optimizor
        opt = self.optimizer([self.U, self.V], lr=self.alpha)

        # training loop
        if self.verbose:
            N = trange(self.n_epoch, ncols=80)
        else:
            N = xrange(self.n_epoch)

        for n in N:
            for u, i, j in batch_prep_neg_sampling(X, Y, self.verbose,
                                                   shuffle=True):

                # flush grad
                opt.zero_grad()

                # forward pass
                l = self.forward(u, i, j)

                # save loss curve
                self.loss_curve.append(l.data)

                # backward pass
                l.backward()

                # update
                opt.step()



if __name__ == "__main__":
    from util import read_data
    from evaluation import *
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    print('Loading data...')
    # d = read_data('./data/ml-100k/u1.base', delimiter='\t')
    # dt = read_data('./data/ml-100k/u1.test',
    #                delimiter='\t', shape=d.shape).tocsr()
    d = read_data('../../../Downloads/playlist_track.csv')
    i, j, v = sp.find(d)
    rnd_idx = np.random.choice(len(i), len(i), replace=False)
    bound = int(len(i) * 0.8)
    rnd_idx_trn = rnd_idx[:bound]
    rnd_idx_val = rnd_idx[bound:]
    d = sp.coo_matrix((v[rnd_idx_trn], (i[rnd_idx_trn], j[rnd_idx_trn])),
                      shape=d.shape)
    dt = sp.coo_matrix((v[rnd_idx_val], (i[rnd_idx_val], j[rnd_idx_val])),
                       shape=d.shape)

    print('Fit model!')
    # fit
    model = BPRMF(10, verbose=True)
    model.fit(d)

    print('Evaluate!')
    # predict
    rprec = []
    ndcg = []
    for u in tqdm(xrange(d.shape[0]), total=d.shape[0], ncols=80):
        true = sp.find(dt[u])[1]
        pred = model.predict(u)

        rprec.append(R_precision(true, pred))
        ndcg.append(NDCG(true, pred))
    rprec = filter(lambda r: r is not None, rprec)
    ndcg = filter(lambda r: r is not None, ndcg)

    print('R Precision: {:.4f}'.format(np.mean(rprec)))
    print('NDCG: {:.4f}'.format(np.mean(ndcg)))

    fig, ax = plt.subplots(1, 1)
    ax.plot(model.loss_curve)
    fig.savefig('./data/loss.png')
