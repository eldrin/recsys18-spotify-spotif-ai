import os
from functools import partial
from collections import namedtuple
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.autograd import Variable

from prefetch_generator import background
from tqdm import tqdm, trange

# from evaluation import r_precision, NDCG
import sys
sys.path.append('../RecsysChallengeTools/')
from metrics import ndcg, r_precision, playlist_extender_clicks
NDCG = partial(ndcg, k=500)

from mfmlp import MPDSampler
from data import get_ngram, get_unique_ngrams

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
            'playlists': './data/playlist_hash_ss.csv',
            'tracks': './data/track_hash_ss.csv',
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
        'learn_rate': 0.001,
        'batch_size': 500,
        'learn_metric': True,
        'non_lin': nn.ReLU,
        'dropout': False,
        'l2': 1e-8,
        'alpha': 0,
        'ngram_n': 3
    },

    'evaluation':{
        'cutoff':500
    }
}


def transform_id2ngram_id(ids, title_dict, ngram_dict, n=3):
    """
    transform a index (either playlist or track)
    into sequence of ngram_id
    """
    out = []
    for i in ids:
        out.append(
            [ngram_dict[w] for w in get_ngram(title_dict[i], n=n)]
        )
    return out


class SeqTensor:
    def __init__(self, seq, title_dict, ngram_dict):
        """"""
        # process seq
        seq_ngram = transform_id2ngram_id(seq, title_dict, ngram_dict)
        lengths = map(len, seq_ngram)
        max_len = max(lengths)
        lengths = torch.cuda.LongTensor(lengths)
        X_pl = torch.cuda.LongTensor(
            [a + [0] * (max_len - len(a)) for a in seq_ngram])
        length_sorted, ind = lengths.sort(descending=True)

        # assign properties
        self.seq = X_pl[ind]
        self.lengths = length_sorted
        self.ind = ind

    def unsort(self, h):
        """"""
        # GPU version
        unsorted = self.ind.new(*self.ind.shape)
        unsorted.scatter_(0, self.ind,
                          torch.cuda.LongTensor(range(self.ind.shape[0])))
        # # CPU version
        # unsorted = map(
        #     lambda x:x[0],
        #     sorted(enumerate(self.ind.tolist()), key=lambda x:x[1])
        # )
        return h[unsorted]


class CFRNN(nn.Module):
    """"""
    def __init__(self, n_components, n_users, n_items,user_emb, item_emb,
                 n_hid=100, user_train=True, item_train=True, n_layers=1,
                 non_lin=nn.ReLU, layer_norm=False, drop_out=0, learn_metric=True):
        """"""
        super(CFRNN, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.non_lin = non_lin
        self.learn_metric = learn_metric
        self.embs = {}

        # setup learnable embedding layers
        for k, emb, n, train in [('user', user_emb, n_users, user_train),
                                 ('item', item_emb, n_items, item_train)]:
            if emb is not None:
                r = emb.shape[-1]
                self.embs[k] = nn.Embedding(n, r)
                self.embs[k].weight.data.copy_(torch.FloatTensor(emb))
            else:
                r = n_components
                self.embs[k] = nn.Embedding(n, r)
            self.embs[k].weight.requires_grad = train
            self.add_module(k, self.embs[k])

        self.user_rnn = nn.GRU(n_components, n_hid, n_layers, batch_first=True)
        self.item_rnn = nn.GRU(n_components, n_hid, n_layers, batch_first=True)

        if learn_metric:
            self.metric = nn.Sequential(
                nn.Linear(n_hid * 2, n_hid),
                self.non_lin(),
                nn.Linear(n_hid, 1)
            )
            self.add_module('metric', self.metric)
            # self.metric = nn.Linear(n_hid * 2, 1)

    def forward(self, pid, tid):
        """
        pid: SeqTensor instance for batch of playlist
        tid: SeqTensor instance for batch of tracks
        """
        # input
        emb_pl = self.embs['user'](Variable(pid.seq))
        emb_tr = self.embs['item'](Variable(tid.seq))

        # pack
        emb_pl = pack_padded_sequence(emb_pl, pid.lengths.tolist(), batch_first=True)
        emb_tr = pack_padded_sequence(emb_tr, tid.lengths.tolist(), batch_first=True)

        # rnn
        out_u, hid_u = self.user_rnn(emb_pl)
        out_i, hid_i = self.item_rnn(emb_tr)

        # unpack & unsort batch order
        hid_u = pid.unsort(hid_u[-1])  # only take last rnn layer
        hid_i = tid.unsort(hid_i[-1])

        # obtain final estimation
        if self.learn_metric:
            # concat state
            h = torch.cat([hid_u, hid_i], dim=-1)  # (batch, n_hid * 2)
            y_pred = self.metric(h)[:, 0]
        else:
            y_pred = torch.sum(hid_u * hid_i, dim=-1)

        return y_pred

    def user_factor(self, pid):
        """"""
        emb_pl = self.embs['user'](Variable(pid.seq))
        emb_pl = pack_padded_sequence(emb_pl, pid.lengths.tolist(), batch_first=True)
        out_u, hid_u = self.user_rnn(emb_pl)
        return pid.unsort(hid_u[-1])

    def item_factor(self, tid):
        """"""
        emb_tr = self.embs['item'](Variable(tid.seq))
        emb_tr = pack_padded_sequence(emb_tr, tid.lengths.tolist(), batch_first=True)
        out_i, hid_i = self.item_rnn(emb_tr)
        return tid.unsort(hid_i[-1])


if __name__ == "__main__":
    """"""
    # setup some aliases
    HP = CONFIG['hyper_parameters']

    # make ngram dicts
    print('Preparing {:d}-Gram dictionary...'.format(HP['ngram_n']))
    uniq_playlists = pd.read_csv(CONFIG['path']['data']['playlists'], sep='\t',
                                 index_col=None, header=None)
    uniq_ngrams_pl = get_unique_ngrams(uniq_playlists[1].values, n=HP['ngram_n'])
    playlist_dict = dict(uniq_playlists[[3, 1]].as_matrix())
    pl_ngram_dict = {v:k for k, v in enumerate(uniq_ngrams_pl)}
    print('Uniq {:d}-Gram for playlists: {:d}'
          .format(HP['ngram_n'], len(uniq_ngrams_pl)))

    uniq_tracks = pd.read_csv(CONFIG['path']['data']['tracks'], sep='\t',
                              index_col=None, header=None)
    uniq_ngrams_tr = get_unique_ngrams(uniq_tracks[1].values, n=HP['ngram_n'])
    track_dict = dict(uniq_tracks[[3, 1]].as_matrix())
    tr_ngram_dict = {v:k for k, v in enumerate(uniq_ngrams_tr)}
    print('Uniq {:d}-Gram for tracks: {:d}'
          .format(HP['ngram_n'], len(uniq_ngrams_tr)))

    # prepare model instances
    sampler = MPDSampler(CONFIG, verbose=True)
    model = CFRNN(
        n_components=128, n_users=len(uniq_ngrams_pl),
        n_items=len(uniq_ngrams_tr),user_emb=None, item_emb=None,
        n_hid=128, user_train=True, item_train=True, n_layers=1,
        non_lin=nn.ReLU, layer_norm=False
    ).cuda()

    # set loss / optimizer
    f_loss = nn.BCEWithLogitsLoss().cuda()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        weight_decay=HP['l2'], lr=HP['learn_rate'])
    # opt = torch.optim.Adagrad(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     weight_decay=HP['l2'], lr=HP['learn_rate'])

    # main training loop
    model.train()
    try:
        epoch = trange(HP['num_epochs'], ncols=80)
        for n in epoch:
            for batch in sampler.generator():
                batch_t = np.array(batch).T

                # parse in / out
                pid, tid = batch_t[:2]
                pref, conf = [
                    torch.cuda.FloatTensor(a) for a in batch_t[-2:]]

                # process seqs
                pid = SeqTensor(pid, playlist_dict, pl_ngram_dict)
                tid = SeqTensor(tid, track_dict, tr_ngram_dict)

                # flush grad
                opt.zero_grad()

                # forward pass
                y_pred = model.forward(pid, tid)

                # calc loss
                l = f_loss(y_pred, Variable(pref))

                # back-propagation
                l.backward()

                # clip gradients
                grad_norm = nn.utils.clip_grad_norm(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=100
                )

                # update
                opt.step()

                # log
                epoch.set_description(
                    '[loss : {:.3f} / |grad|: {:.2f}]'.format(float(l.data), grad_norm)
                )

    except KeyboardInterrupt:
        print('[Warning] User stopped the training!')
    # switch off to evaluation mode
    model.eval()

    print('Evaluate!')
    # fetch testing playlists
    trg_u = sampler.test['playlist'].unique()

    # 1) extract playlist / track factors first from embedding-rnn blocks
    b = 500
    m = model.n_items
    tid = range(m)
    #   1.1) ext item factors first
    Q = []
    for j in trange(0, m, b, ncols=80):
        tid_ = SeqTensor(tid[j:j+b], track_dict, tr_ngram_dict)
        Q.append(model.item_factor(tid_).data.cpu().numpy())
    Q = np.concatenate(Q, axis=0)

    #   1.2) ext playlist factors
    P = []
    for j in trange(0, len(trg_u), b, ncols=80):
        pid_ = SeqTensor(trg_u[j:j+b], playlist_dict, pl_ngram_dict)
        P.append(model.user_factor(pid_).data.cpu().numpy())
    P = np.concatenate(P, axis=0)

    #   2) calculate scores using the same way of MLP case
    rprec = []
    ndcg = []
    for j, u in tqdm(enumerate(trg_u), total=len(trg_u), ncols=80):
        true = sampler.test[sampler.test['playlist'] == u]['track']
        true_t = set(sampler.train[sampler.train['playlist'] == u]['track'])

        # predict k
        cat = np.concatenate([np.tile(P[j], (Q.shape[0], 1)), Q], axis=-1)
        pred = []
        for k in trange(0, m, b, ncols=80):
            pred.append(model.metric(Variable(torch.cuda.FloatTensor(cat[k:k+b])))[:, 0].data)
        pred = np.concatenate(pred, axis=0)
        ind = np.argsort(pred)[::-1][:1000]

        # exclude training data
        pred = filter(lambda x: x not in true_t, ind)[:500]

        rprec.append(r_precision(true, pred))
        ndcg.append(NDCG(true, pred))
    rprec = filter(lambda r: r is not None, rprec)
    ndcg = filter(lambda r: r is not None, ndcg)

    print('R Precision: {:.4f}'.format(np.mean(rprec)))
    print('NDCG: {:.4f}'.format(np.mean(ndcg)))
