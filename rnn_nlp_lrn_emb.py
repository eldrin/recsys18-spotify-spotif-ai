import os
import re
from functools import partial
from itertools import chain
from collections import namedtuple
import cPickle as pkl
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from nltk.tokenize.nist import NISTTokenizer

import torch
from torch import optim
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.autograd import Variable

from prefetch_generator import background
from tqdm import tqdm, trange

# from evaluation import r_precision, NDCG
import sys
sys.path.append(os.path.join(os.getcwd(), 'RecsysChallengeTools/'))
from metrics import ndcg, r_precision, playlist_extender_clicks
NDCG = partial(ndcg, k=500)

from mfmlp import MPDSampler, NEGCrossEntropyLoss
from data import get_ngram, get_unique_ngrams
from pretrain_word2vec import load_n_process_data
from util import read_hash, MultipleOptimizer
from nlp_rnn_factor_learn import CONFIG

try:
    print(torch.cuda.current_device())
    floatX = torch.cuda.FloatTensor
except:
    floatX = torch.FloatTensor


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
    def __init__(self, seq, title_dict=None, ngram_dict=None):
        """"""
        # process seq
        if title_dict is None and ngram_dict is None:
            seq_ngram = seq
        else:  # process on-the-go
            seq_ngram = transform_id2ngram_id(seq, title_dict, ngram_dict)
        lengths = map(len, seq_ngram)
        max_len = max(lengths)
        lengths = torch.cuda.LongTensor(lengths)
        X_pl = torch.cuda.LongTensor(
            [a + [0] * (max_len - len(a)) for a in seq_ngram])
        length_sorted, ind = lengths.sort(descending=True)
        _, self.unsort_ind = ind.sort()

        # assign properties
        self.seq = X_pl[ind]
        self.lengths = length_sorted
        self.ind = ind

    def unsort(self, h):
        """"""
        return h[self.unsort_ind]


class UserRNN(nn.Module):
    """"""
    def __init__(self, n_components, n_users, n_hid=100, n_out=16,
                 user_train=True, n_layers=1, non_lin=nn.ReLU, layer_norm=False,
                 drop_out=0, learn_metric=True, sparse_embedding=True):
        """"""
        super(UserRNN, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_layers = n_layers
        self.non_lin = non_lin
        self.n_out = n_out
        self.learn_metric = learn_metric
        self.embs = {}

        # setup learnable embedding layers
        self.emb = nn.Embedding(
            n_users, n_components, sparse=sparse_embedding)
        self.emb.weight.requires_grad = user_train
        self.user_rnn = nn.LSTM(n_components, n_hid, n_layers,
                                batch_first=True)
        self.user_out = nn.Linear(n_hid, n_out)

    def forward(self, pid):
        """
        pid: SeqTensor instance for batch of playlist
        """
        # process seqs
        pid = SeqTensor(pid, None, None)

        # process rnn
        emb_pl = self.emb(Variable(pid.seq))
        emb_pl = pack_padded_sequence(emb_pl, pid.lengths.tolist(), batch_first=True)
        out_u, hid_u = self.user_rnn(emb_pl)

        # unpack & unsort batch order
        hid_u = pid.unsort(hid_u[0][-1])  # only take last rnn layer

        # obtain final estimation
        out_u = self.user_out(hid_u)
        return out_u

    def user_factor(self, pid):
        """"""
        pid = SeqTensor(pid, None, None)
        emb_pl = self.emb(Variable(pid.seq))
        emb_pl = pack_padded_sequence(emb_pl, pid.lengths.tolist(), batch_first=True)
        out_u, hid_u = self.user_rnn(emb_pl)
        return self.user_out(pid.unsort(hid_u[0][-1]))


if __name__ == "__main__":
    """"""
    # setup some aliases
    HP = CONFIG['hyper_parameters']

    # make ngram dicts
    print('Preparing {:d}-Gram dictionary...'.format(HP['ngram_n']))
    uniq_playlists = read_hash(CONFIG['path']['data']['playlists'])
    uniq_playlists.loc[:, 0] = uniq_playlists[0].astype(int)
    uniq_playlists.loc[:, 3] = uniq_playlists[3].astype(int)

    uniq_ngrams_pl = get_unique_ngrams(uniq_playlists[1].values, n=HP['ngram_n'])
    playlist_dict_ = dict(uniq_playlists[[3, 1]].values)
    pl_ngram_dict = {v:k for k, v in enumerate(uniq_ngrams_pl)}

    playlist_dict = dict(zip(
        playlist_dict_.keys(),
        transform_id2ngram_id(
            playlist_dict_.keys(), playlist_dict_, pl_ngram_dict, n=HP['ngram_n'])
    ))

    print('Uniq {:d}-Gram for playlists: {:d}'
          .format(HP['ngram_n'], len(uniq_ngrams_pl)))

    # prepare model instances
    sampler = MPDSampler(CONFIG, verbose=True)
    model = UserRNN(
        n_components=HP['n_embedding'],
        n_users=len(uniq_ngrams_pl),
        n_hid=HP['n_hid'], n_out=HP['n_out_embedding'],
        user_train=True, n_layers=1, non_lin=HP['non_lin'],
        layer_norm=False, drop_out=0, learn_metric=True,
        sparse_embedding=True
    ).cuda()

    # set loss / optimizer
    if HP['loss'] == 'MSE':
        # f_loss = nn.SmoothL1Loss().cuda()
        f_loss = nn.MSELoss().cuda()

        # load target (pre-trained) playlist factors
        playlist_factors = np.load(
            CONFIG['path']['embeddings']['U']).astype(np.float32)

    elif HP['loss'] == 'SGNS':
        f_loss = NEGCrossEntropyLoss().cuda()

        # load target (pre-trained) playlist factors
        track_factors = np.load(
            CONFIG['path']['embeddings']['V']).astype(np.float32)

    elif HP['loss'] == 'all':
        # load target (pre-trained) playlist factors
        playlist_factors = np.load(
            CONFIG['path']['embeddings']['U']).astype(np.float32)

        # load target (pre-trained) playlist factors
        track_factors = np.load(
            CONFIG['path']['embeddings']['V']).astype(np.float32)

        # setup total loss
        # mse = nn.SmoothL1Loss().cuda()
        mse = nn.MSELoss().cuda()
        sgns = NEGCrossEntropyLoss().cuda()
        def f_loss(h, v, pref, p, conf=None, coeff=0.5):
            """"""
            hv = torch.bmm(
                v, h.view(y_pred.shape[0], y_pred.shape[-1], 1)
            ).squeeze()
            return coeff * sgns(hv, pref, conf) + (1.-coeff) * mse(h, p)


    sprs_prms = map(
        lambda x: x[1],
        filter(lambda kv: kv[0] in {'emb.weight'},
               model.named_parameters())
    )
    dnse_prms = map(
        lambda x: x[1],
        filter(lambda kv: kv[0] not in {'emb.weight'},
               model.named_parameters())
    )
    opt = MultipleOptimizer(
        optim.SparseAdam(sprs_prms, lr=HP['learn_rate']),
        optim.Adam(dnse_prms, lr=HP['learn_rate'], amsgrad=True)
    )

    # main training loop
    model.train()
    losses = []
    try:
        epoch = trange(HP['num_epochs'], ncols=80)
        for n in epoch:
            for batch in sampler.generator():

                # parse in / out
                pid_ = [x[0] for x in batch]  # (n_batch,)
                pid = [playlist_dict[i] for i in pid_]
                conf_ = [x[-1] for x in batch]
                conf = Variable(torch.FloatTensor(conf_).cuda())

                # flush grad
                opt.zero_grad()

                # forward pass
                y_pred = model.forward(pid)

                if HP['loss'] == 'MSE':
                    y_true = Variable(
                        torch.from_numpy(playlist_factors[pid_]).cuda()
                    )
                    # calc loss
                    l = f_loss(y_pred, y_true)

                elif HP['loss'] == 'SGNS':
                    tid_ = [[x[1]] + x[2] for x in batch]
                    pref_ = [[1.] + [-1.] * len(x[2]) for x in batch]
                    pref = Variable(torch.FloatTensor(pref_).cuda())

                    # calc loss
                    v = Variable(
                        torch.from_numpy(
                            np.array([track_factors[t] for t in tid_])
                        ).cuda()
                    )
                    hv = torch.bmm(
                        v, y_pred.view(y_pred.shape[0], y_pred.shape[-1], 1)
                    ).squeeze()
                    l = f_loss(hv, pref) + ((y_pred)**2).sum()**.5

                elif HP['loss'] == 'all':
                    tid_ = [[x[1]] + x[2] for x in batch]
                    pref_ = [[1.] + [-1.] * len(x[2]) for x in batch]
                    pref = Variable(torch.FloatTensor(pref_).cuda())
                    y_true = Variable(
                        torch.from_numpy(playlist_factors[pid_]).cuda()
                    )
                    v = Variable(
                        torch.from_numpy(
                            np.array([track_factors[t] for t in tid_])
                        ).cuda()
                    )

                    # calc loss
                    l = f_loss(y_pred, v, pref, y_true)

                # back-propagation
                l.backward()

                # update
                opt.step()

                # log
                if HP['loss'] == 'MSE':
                    epoch.set_description(
                        '[loss : {:.4f}]'.format(float(l.data) * HP['batch_size'])
                    )
                elif HP['loss'] == 'SGNS' or HP['loss'] == 'all':
                    epoch.set_description(
                        '[loss : {:.4f}]'.format(float(l.data))
                    )
                losses.append(float(l.data))

    except KeyboardInterrupt:
        print('[Warning] User stopped the training!')
    # switch off to evaluation mode
    model.eval()

    # 1) extract playlist / track factors first from embedding-rnn blocks
    #   1.1) ext item factors first
    Q = track_factors

    #   1.2) ext playlist factors
    b = 500
    n = uniq_playlists.shape[0]
    pid = range(n)
    P = []
    for j in trange(0, n+(n%b), b, ncols=80):
        if j > n:
            continue
        pid_ = [playlist_dict[jj] for jj in pid[j:j+b]]
        P.append(model.user_factor(pid_).data.cpu().numpy())
    P = np.concatenate(P, axis=0).astype(np.float32)
    np.save('./data/models/title_rnn_U.npy', P)

    if sampler.test is not None:
        print('Evaluate!')
        # fetch testing playlists
        y, yt = sampler.train, sampler.test
        trg_u = yt['playlist'].unique()
        yt_tracks = yt.groupby('playlist')['track'].apply(list)
        y_tracks = y[y['value']==1].groupby('playlist')['track'].apply(set)

        #   2) calculate scores using the same way of MLP case
        rprec = []
        ndcg = []
        for j, u in tqdm(enumerate(trg_u), total=len(trg_u), ncols=80):
            true_ts = yt_tracks.loc[u]
            if u in y_tracks.index:
                true_tr = y_tracks.loc[u]
            else:
                true_tr = set()

            # predict k
            pred = -P[u].dot(Q.T)
            ind = np.argpartition(pred, 650)[:650]
            ind = ind[pred[ind].argsort()]

            # exclude training data
            pred = filter(lambda x: x not in true_tr, ind)[:500]

            rprec.append(r_precision(true_ts, pred))
            ndcg.append(NDCG(true_ts, pred))
        rprec = filter(lambda r: r is not None, rprec)
        ndcg = filter(lambda r: r is not None, ndcg)

        print('R Precision: {:.4f}'.format(np.mean(rprec)))
        print('NDCG: {:.4f}'.format(np.mean(ndcg)))

    # 3) print out the loss evolution as a image
    plt.plot(losses)
    plt.savefig('./data/losses.png')
