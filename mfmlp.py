import os
import random
from functools import partial
from itertools import chain

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
        'eval_while_fit': True,
        'sample_weight': False,
        'sample_weight_power': 3./4,
        'sample_threshold': 1e-6,
        'num_epochs': 100,
        'neg_sample': 10,
        'learn_rate': 0.001,
        'batch_size': 128,
        'mlp_arch': [],
        'learn_metric': False,
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
        dict(dat['artist2track'][[1, 0]].values)
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
        self.is_weight = config['hyper_parameters']['sample_weight']
        self.weight_pow = config['hyper_parameters']['sample_weight_power']
        self.threshold = config['hyper_parameters']['sample_threshold']

        # prepare positive sample pools
        self.pos_tracks = dict(self.triplet.groupby('playlist')['track'].apply(set))
        self.pos_tracks_t = dict(self.test.groupby('playlist')['track'].apply(set))
        if self.is_weight:
            # check word (track) frequency
            track_count = self.triplet.groupby('track').count()['playlist']
            f_w = np.array(map(
                lambda x: x[1],
                sorted(track_count.to_dict().items(), key=lambda x: x[0])
            ))

            # Sub-sampling
            p_drop = dict(zip(range(len(f_w)), 1. - np.sqrt(self.threshold / f_w)))
            train_words = filter(lambda r: np.random.random() < (1 - r[1]), p_drop.items())
            train_words = map(lambda r: r[0], train_words)

            # preprocess weighted sampling pool
            f_w_pow = np.round(np.power(f_w, self.weight_pow))[train_words]
            self.items = list(
                chain.from_iterable([[k] * int(v) for k, v in zip(train_words, f_w_pow)])
            )
            self.items = dict(enumerate(self.items))

        else:
            self.items = range(self.n_tracks)
            self.items = dict(enumerate(self.items))

        self.neg = config['hyper_parameters']['neg_sample']
        self.verbose = verbose

    @background(max_prefetch=10000)
    def generator(self):
        """"""
        if self.verbose:
            M = tqdm(self.triplet.sample(frac=1).values, ncols=80)
        else:
            M = self.triplet.sample(frac=1).values

        batch = []
        # s_pos = 240.7895  # log_surplus(1)
        s_pos = 1
        s_neg = 1
        # negs = set()
        for u, i, v in M:
            # positive sample / yield
            pos_i = self.pos_tracks[u]
            if v == 0:
                continue
            batch.append((u, i, self.track2artist[i], 1, s_pos))

            # draw negative samples (for-loop)
            for k in xrange(self.neg):
                # j_ = np.random.choice(self.n_tracks)
                j_ = self.items[np.random.choice(len(self.items))]
                # while (j_ in pos_i) or (j_ in negs):
                while j_ in pos_i:
                    # j_ = np.random.choice(self.n_tracks)
                    j_ = self.items[np.random.choice(len(self.items))]
                # negs.update((j_,))

                # negtive sample has 0 interaction (conf==1)
                batch.append((u, j_, self.track2artist[j_], -1, s_neg))

            # batch.append(batch_)
            if len(batch) >= self.batch_size * (1. + self.neg):
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
                 non_lin=nn.ReLU, batch_norm=True, drop_out=0, learn_metric=True,
                 init=0.01):
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
                u = np.random.randn(n, r) * init
                self.embs[k] = nn.Embedding(n, r)
                self.embs[k].weight.data.copy_(torch.FloatTensor(u))
            self.embs[k].weight.requires_grad = train

            # get non-linear embedding
            n_in = r
            layers = [self.embs[k]]
            for l, n_h in enumerate(self.arch[1:]):
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
        p = self.mlps['user'](u)
        q = self.mlps['item'](i)

        if self.learn_metric:
            elem_sum = p * q
            return self.metric(elem_sum)[:, 0]
        else:
            y = torch.bmm(
                p.view(p.shape[0], 1, p.shape[1]),
                q.view(q.shape[0], q.shape[1], 1)
            ).squeeze(1).squeeze(1)
            return y


class GMFMLP(nn.Module):
    """ main booster model """
    def __init__(self, n_components, n_users, n_items, alpha=0.5,
                 architecture=[50, 50, 50],
                 user_train=True, item_train=True, attr_train=False,
                 user_emb=None, item_emb=None, attr_emb=None,
                 non_lin=nn.ReLU, batch_norm=True, drop_out=True):
        """"""
        super(GMFMLP, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_items = n_items
        self.arch = architecture
        self.arch.append(self.n_components)
        self.drop_out = drop_out
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

        self.dropout = nn.Dropout()  # only for emb?

        self.metric_mlp = nn.Linear(n_components, 1)
        self.metric_mf = nn.Linear(n_components, 1)

    def forward(self, pid, tid):
        """"""
        # input
        emb_pl, emb_tr = self.emb_u(pid), self.emb_i(tid)
        emb = torch.cat([self.embs['user'](pid), self.embs['item'](tid)], dim=-1)

        # if self.drop_out:
        #     emb = self.dropout(emb)

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


class NEGCrossEntropyLoss(nn.Module):
    """"""
    def __init__(self):
        """"""
        super(NEGCrossEntropyLoss, self).__init__()

    def forward(self, hv, targ, weight=None):
        """"""
        # targ[targ==0] = -1
        if weight is not None:
            return -(F.logsigmoid(hv * targ) * weight).mean()
        else:
            return -F.logsigmoid(hv * targ).mean()


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
        self.eval_while_fit = config['hyper_parameters']['eval_while_fit']
        self.verbose = verbose
        self.embs = _load_embeddings(config)
        self.loss_curve = []
        self.acc_curve = {'ndcg':[], 'rprec':[]}

        # build mlp
        # load embedding and put them in Embedding layer
        P, V, W = [
            torch.Tensor(self.embs[a]) for a in ['U', 'V', 'W']]

        # # make concatenated item factor
        # W = W[[self.track2artist[t] for t in xrange(V.shape[0])]]
        # Q = np.concatenate([W, X], axis=-1)

        # initiate model
        self.core_model = CFMLP(
            # n_components=P.shape[-1],
            n_components=10,
            n_users=P.shape[0], n_items=V.shape[0],
            architecture=self.arch,
            user_train=True, item_train=True,
            user_emb=None, item_emb=None,
            non_lin=self.non_lin, batch_norm=False,
            learn_metric=False
        ).cuda()

        # self.core_model = GMFMLP(
        #     n_components=P.shape[-1],
        #     n_users=P.shape[0], n_items=V.shape[0],
        #     architecture=self.arch, alpha=self.alpha,
        #     user_train=True, item_train=True,
        #     user_emb=None, item_emb=None,
        #     non_lin=self.non_lin, batch_norm=True,
        #     drop_out=False
        # ).cuda()

        # setup loss function
        # self.loss_fn = nn.MSELoss(reduce=False).cuda()
        # self.loss_fn = nn.BCEWithLogitsLoss().cuda()
        self.loss_fn = NEGCrossEntropyLoss().cuda()

        self.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.core_model.parameters()),
            weight_decay=self.l2, lr=self.lr, eps=1e-8
        )
        # self.optim = torch.optim.SGD(
        #     filter(lambda p: p.requires_grad, self.core_model.parameters()),
        #     weight_decay=self.l2, lr=self.lr)

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
        y_pred = self.predict(Variable(pid), Variable(tid))

        # calculate loss
        # # get loss (weighted)
        # l = torch.sum(
        #     self.loss_fn(y_pred, Variable(preference)) * Variable(confidence)
        # )
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
                    pid = torch.cuda.LongTensor(batch_t[0])
                    tid = torch.cuda.LongTensor(batch_t[1])
                    pref = torch.cuda.FloatTensor(batch_t[3])
                    conf = torch.cuda.FloatTensor(batch_t[4])
                    # pid = torch.cuda.LongTensor(map(lambda x: x[0], batch))
                    # tid = torch.cuda.LongTensor(map(lambda x: x[1], batch))
                    # pref = torch.cuda.FloatTensor(map(lambda x: x[3], batch))
                    # conf = torch.cuda.FloatTensor(map(lambda x: x[4], batch))

                    loss = self.partial_fit(pid, tid, pref, conf)

                    if self.verbose:
                        self.loss_curve.append(float(loss))
                        epoch.set_description(
                            '[loss : {:.4f}]'.format(float(loss))
                        )

                if self.eval_while_fit:
                    trg_u = sampler.test['playlist'].unique()
                    rprec = []
                    ndcg = []
                    for u in trg_u:
                        true = sampler.pos_tracks_t[u]
                        true_t = sampler.pos_tracks[u]
                        pred = model.predict_k(u, k=500 * 2)
                        pred = filter(lambda x: x not in true_t, pred)[:500]
                        rprec.append(r_precision(list(true), pred))
                        ndcg.append(NDCG(list(true), pred))
                    rprec = filter(lambda r: r is not None, rprec)
                    ndcg = filter(lambda r: r is not None, ndcg)
                    self.acc_curve['rprec'].append(np.mean(rprec))
                    self.acc_curve['ndcg'].append(np.mean(ndcg))

        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')
        self.core_model.eval()
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.loss_curve)
        axs[1].plot(self.acc_curve['rprec'])
        axs[1].plot(self.acc_curve['ndcg'])
        fig.savefig('./data/loss.png')


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
        true = sampler.pos_tracks_t[u]
        true_t = sampler.pos_tracks[u]
        pred = model.predict_k(u, k=K * 2)

        # exclude training data
        pred = filter(lambda x: x not in true_t, pred)[:K]

        rprec.append(r_precision(list(true), pred))
        ndcg.append(NDCG(list(true), pred))
    rprec = filter(lambda r: r is not None, rprec)
    ndcg = filter(lambda r: r is not None, ndcg)

    print('R Precision: {:.4f}'.format(np.mean(rprec)))
    print('NDCG: {:.4f}'.format(np.mean(ndcg)))
