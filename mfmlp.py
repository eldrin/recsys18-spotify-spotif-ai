import os
from functools import partial
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from prefetch_generator import background
from tqdm import tqdm, trange

# from evaluation import r_precision, NDCG
import sys
sys.path.append('../RecsysChallengeTools/')
from metrics import ndcg, r_precision, playlist_extender_clicks
NDCG = partial(ndcg, k=500)

try:
    print(torch.cuda.current_device())
    floatX = torch.cuda.FloatTensor
except:
    floatX = torch.FloatTensor


CONFIG = {
    'path':{
        'embeddings':{
            'U': './data/bpr_U.npy',
            'V': './data/bpr_V.npy',
            'W': './data/bpr_W.npy',
            'X': './data/spotify_feature_popularity_scaled_ss2.npy'
        },
        'data':{
            'train': './data/playlist_track_ss_train.csv',
            'test': './data/playlist_track_ss_test.csv',
            'artist2track': './data/artist_track_ss.csv',
        },
        'model_out': './models/',
        'log_out': './logs/'
    },

    'hyper_parameters':{
        'num_epochs': 50,
        'neg_sample': 5,
        'learn_rate': 0.01,
        'batch_size': 512,
        'mlp_arch': [64, 64, 32],
        'learn_metric': True,
        'non_lin': nn.ReLU,
        'dropout': False,
        'l2': 1e-8,
        'alpha': 0
    },

    'evaluation':{
        'cutoff':500
    }
}


def _load_embeddings(config):
    """"""
    embs = {
        k: np.load(fn) for k, fn
        in config['path']['embeddings'].iteritems()
    }
    return embs


def _load_data(config):
    """ returns main tuples and track-artist map """
    dat = {
        k: pd.read_csv(fn, header=None) for k, fn
        in config['path']['data'].iteritems()
        if k != 'playlists' and k != 'tracks'
    }
    # dat['main'].columns = ['playlist', 'track']
    dat['train'].columns = ['playlist', 'track', 'value']  # set columns name
    dat['test'].columns = ['playlist', 'track', 'value']
    # return (
    #     dat['main'], dat['train'], dat['test'],
    #     dict(dat['artist2track'][[1, 0]].as_matrix())
    # )
    return (
        None, dat['train'], dat['test'],
        dict(dat['artist2track'][[1, 0]].as_matrix())
    )


def log_surplus(x, alpha=100, eps=0.1):
    """"""
    return 1. + alpha * np.log(1. + (x / eps))


class MPDSampler:
    """"""
    def __init__(self, config, verbose=False):
        """"""
        _, self.train, self.test, self.track2artist = _load_data(config)
        self.triplet = self.train

        self.n_playlists = self.train['playlist'].nunique()
        self.n_tracks = self.train['track'].nunique()
        self.num_interactions = self.train.shape[0]
        self.batch_size = config['hyper_parameters']['batch_size']

        # prepare positive sample pools
        self.pos_tracks = dict(self.triplet.groupby('playlist')['track'].apply(list))
        # self.pos_tracks = {}
        # for u in trange(self.n_playlists, ncols=80):
        #     self.pos_tracks[u] = set(self.triplet[self.triplet['playlist'] == u]['track'])

        self.neg = config['hyper_parameters']['neg_sample']
        self.verbose = verbose

    @background(max_prefetch=10000)
    def generator(self):
        """"""
        dat = self.triplet.values

        if self.verbose:
            M = tqdm(shuffle(dat), ncols=80)
        else:
            M = shuffle(dat)

        batch = []
        for u, i, v in M:  # draw playlist and track (positive)
            # positive sample / yield
            pos_i = self.pos_tracks[u]
            if v == 0:
                continue
            # s = 240.7895  # log_surplus(1)
            s = self.neg
            batch.append((u, i, self.track2artist[i], 1, s))
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

            # sample negative sample / yield
            for _ in xrange(self.neg):
                j_ = np.random.choice(self.n_tracks)
                while j_ in pos_i:
                    j_ = np.random.choice(self.n_tracks)

                # negtive sample has 0 interaction (conf==1)
                batch.append((u, j_, self.track2artist[j_], 0, 1))
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

class CFMLP(nn.Module):
    """
    It provides model such:
        r_{u, i} = P_{u}Q_{i}'
        P_{u} = f(u)
        Q_{i} = g(i)
        f(u) := a_l(W_l\cdot{a_{l-1}(W_{l-1} \dots a_{1}(W_{1}\cdot(E^{U}_{u})) \dots )
        g(i) := similar to f(u)
    """
    def __init__(self, n_components, n_users, n_items, architecture=[50, 50, 50],
                 user_train=True, item_train=True, user_emb=None, item_emb=None,
                 non_lin=nn.ReLU, batch_norm=True, drop_out=0, learn_metric=True):
        """"""
        super(CFMLP, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_items = n_items
        self.arch = architecture
        self.arch.append(self.n_components)
        self.non_lin = non_lin
        self.learn_metric = learn_metric
        self.embs = {}
        self.mlps = {}

        # setup embedding layers
        for k, emb, n, train in [('user', user_emb, n_users, user_train),
                                 ('item', item_emb, n_items, item_train)]:
            if emb is not None:
                r = emb.shape[-1]
                self.embs[k] = nn.Embedding(n, r)
                self.embs[k].weight.data.copy_(torch.FloatTensor(emb))
            else:
                r = architecture[0]
                self.embs[k] = nn.Embedding(n, r)
            self.embs[k].weight.requires_grad = train

            # get non-linear embedding
            n_in = r
            layers = [self.embs[k]]
            for l, n_h in enumerate(self.arch):
                layers.append(nn.Linear(n_in, n_h))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(n_h))
                if l < len(self.arch)-1:
                    layers.append(self.non_lin())
                n_in = n_h  # update input dim
            self.mlps[k] = nn.Sequential(*layers)
            self.add_module(k, self.mlps[k])

        if self.learn_metric:
            self.metric = nn.Linear(self.n_components, 1)

    def forward(self, u, i):
        """"""
        elem_sum = self.mlps['user'](u) * self.mlps['item'](i)
        if self.learn_metric:
            return self.metric(elem_sum)[:, 0]
        else:
            return torch.sum(elem_sum, dim=1)


class GMFMLP(nn.Module):
    """ main booster model """
    def __init__(self, n_components, n_users, n_items, alpha=0.5,
                 architecture=[50, 50, 50],
                 user_train=True, item_train=True, attr_train=False,
                 user_emb=None, item_emb=None, attr_emb=None,
                 non_lin=nn.ReLU, batch_norm=True, drop_out=0):
        """"""
        super(GMFMLP, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_items = n_items
        self.arch = architecture
        self.arch.append(self.n_components)
        self.alpha = alpha
        self.non_lin = non_lin
        self.embs = {}

        # setup pre-trained embedding layers (only for user and item)
        self.emb_u = nn.Embedding(n_users, n_components)
        if user_emb is not None:
            self.emb_u.weight.data.copy_(torch.FloatTensor(user_emb))
        self.emb_i = nn.Embedding(n_items, n_components)
        if item_emb is not None:
            self.emb_i.weight.data.copy_(torch.FloatTensor(item_emb))

        # setup learnable embedding layers
        for k, emb, n, train in [('user', user_emb, n_users, user_train),
                                 ('item', item_emb, n_items, item_train)]:
            if emb is not None:
                r = emb.shape[-1]
                self.embs[k] = nn.Embedding(n, r)
                self.embs[k].weight.data.copy_(torch.FloatTensor(emb))
            else:
                r = architecture[0]
                self.embs[k] = nn.Embedding(n, r)
            self.embs[k].weight.requires_grad = train
            self.add_module(k, self.embs[k])

        # get non-linear embedding
        n_in = sum([emb.embedding_dim for emb in self.embs.values()])
        layers = []
        for l, n_h in enumerate(self.arch):
            layers.append(nn.Linear(n_in, n_h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_h))
            if l < len(self.arch)-1:
                layers.append(self.non_lin())
            n_in = n_h  # update input dim
        self.mlp = nn.Sequential(*layers)
        self.add_module('mlp', self.mlp)

        self.metric_mlp = nn.Linear(n_components, 1)
        self.metric_mf = nn.Linear(n_components, 1)

    def forward(self, pid, tid):
        """"""
        # input
        emb_pl, emb_tr = self.emb_u(pid), self.emb_i(tid)
        emb = torch.cat([self.embs['user'](pid), self.embs['item'](tid)], dim=-1)

        # pass to MLP
        h_mlp = self.mlp(emb)

        # get MF score
        h_mf = emb_pl * emb_tr

        # obtain final estimation
        y_pred = (
            self.alpha * self.metric_mf(h_mf) +
            (1-self.alpha) * self.metric_mlp(h_mlp)
        )

        return y_pred[:, 0]

class RecNet:
    """"""
    def __init__(self, config, verbose=False):
        """"""
        self.track2artist = _load_data(config)[-1]
        self.arch = config['hyper_parameters']['mlp_arch']
        self.lr = config['hyper_parameters']['learn_rate']
        self.num_epochs = config['hyper_parameters']['num_epochs']
        self.non_lin = config['hyper_parameters']['non_lin']
        self.l2 = config['hyper_parameters']['l2']
        self.alpha = config['hyper_parameters']['alpha']
        self.verbose = verbose
        self.embs = _load_embeddings(config)

        # build mlp
        # load embedding and put them in Embedding layer
        P, V, W, X = [
            torch.Tensor(self.embs[a]) for a in ['U', 'V', 'W', 'X']]

        # make concatenated item factor
        W = W[[self.track2artist[t] for t in xrange(V.shape[0])]]
        Q = np.concatenate([W, X], axis=-1)

        # # initiate model
        # self.core_model = CFMLP(
        #     n_components=self.arch[-1],
        #     n_users=P.shape[0], n_items=V.shape[0],
        #     architecture=self.arch[:-1],
        #     user_train=True, item_train=True,
        #     user_emb=None, item_emb=None,
        #     non_lin=self.non_lin, batch_norm=False,
        #     learn_metric=True
        # ).cuda()

        self.core_model = GMFMLP(
            n_components=P.shape[-1],
            n_users=P.shape[0], n_items=V.shape[0],
            architecture=self.arch, alpha=self.alpha,
            user_train=True, item_train=True,
            user_emb=None, item_emb=None,
            non_lin=self.non_lin, batch_norm=False,
        ).cuda()


        # setup loss function
        # self.loss_fn = nn.MSELoss(reduce=False).cuda()
        self.loss_fn = nn.BCEWithLogitsLoss().cuda()

        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.core_model.parameters()),
            weight_decay=self.l2, lr=self.lr
        )

    def predict(self, pid, tid):
        """"""
        return self.core_model.forward(pid, tid)

    def predict_k(self, pid, k=500):
        """"""
        pred = []
        b = 10000
        m = self.core_model.n_items
        pid = np.ones(m) * pid
        tid = range(m)
        for j in range(0, m, b):
            pred.append(
                self.predict(
                    Variable(torch.cuda.LongTensor(pid[j:j+b])),
                    Variable(torch.cuda.LongTensor(tid[j:j+b])),
                ).data
            )
        pred = np.concatenate(pred)
        return np.argsort(pred)[::-1][:k]

    def partial_fit(self, pid, tid, preference, confidence):
        """"""
        # update
        self.optim.zero_grad()  # flush last grad

        # forward pass
        y_pred = self.predict(pid, tid)

        # calculate loss
        # # get loss (weighted MSE)
        # l = torch.sum(
        #     self.loss_fn(y_pred, Variable(preference)) * Variable(confidence)
        # )

        # get loss (MSE) or (BCE)
        l = self.loss_fn(y_pred, Variable(preference))

        l.backward()  # back-propagation

        self.optim.step()

        return l.data

    def fit(self, sampler):
        """"""
        self.core_model.train()
        # training loop
        if self.verbose:
            epoch = trange(self.num_epochs, ncols=80)
        else:
            epoch = xrange(self.num_epochs)
        try:
            for n in epoch:
                for batch in sampler.generator():
                    batch_t = np.array(batch).T

                    pid, tid = [
                        torch.cuda.LongTensor(a) for a in batch_t[:2]]
                    pref, conf = [
                        torch.cuda.FloatTensor(a) for a in batch_t[-2:]]

                    loss = self.partial_fit(pid, tid, pref, conf)

                    if self.verbose:
                        epoch.set_description(
                            '[loss : {:.4f}]'.format(float(loss))
                        )
        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')
        self.core_model.eval()


if __name__ == "__main__":
    K = CONFIG['evaluation']['cutoff']
    sampler = MPDSampler(CONFIG, verbose=True)
    # model = MFMLP(CONFIG)
    model = RecNet(CONFIG, verbose=True)
    model.fit(sampler)

    print('Evaluate!')
    # predict
    trg_u = sampler.test['playlist'].unique()
    rprec = []
    ndcg = []
    for u in tqdm(trg_u, total=len(trg_u), ncols=80):
        true = sampler.test[sampler.test['playlist'] == u]['track']
        true_t = set(sampler.train[sampler.train['playlist'] == u]['track'])
        pred = model.predict_k(u, k=K * 2)

        # exclude training data
        pred = filter(lambda x: x not in true_t, pred)[:K]

        rprec.append(r_precision(true, pred))
        ndcg.append(NDCG(true, pred))
    rprec = filter(lambda r: r is not None, rprec)
    ndcg = filter(lambda r: r is not None, ndcg)

    print('R Precision: {:.4f}'.format(np.mean(rprec)))
    print('NDCG: {:.4f}'.format(np.mean(ndcg)))
