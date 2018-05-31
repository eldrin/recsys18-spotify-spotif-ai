import os
import re
from functools import partial
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
from cfrnn_conf_insy import CONFIG

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
        # GPU version
        # unsorted = self.ind.new(*self.ind.shape)
        # unsorted.scatter_(0, self.ind,
        #                   torch.cuda.LongTensor(range(self.ind.shape[0])))
        # # CPU version
        # unsorted = map(
        #     lambda x:x[0],
        #     sorted(enumerate(self.ind.tolist()), key=lambda x:x[1])
        # )
        # return h[unsorted]
        return h[self.unsort_ind]


class CFRNN(nn.Module):
    """"""
    def __init__(self, n_components, n_users, n_items,user_emb, item_emb,
                 n_hid=100, n_out=16, user_train=True, item_train=True, n_layers=1,
                 non_lin=nn.ReLU, layer_norm=False, drop_out=0, learn_metric=True):
        """"""
        super(CFRNN, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.non_lin = non_lin
        self.n_out = n_out
        self.learn_metric = learn_metric
        self.embs = {}

        # setup learnable embedding layers
        for k, emb, n, train in [('user', user_emb, n_users, user_train),
                                 ('item', item_emb, n_items, item_train)]:
            if emb is not None:
                r = emb.shape[-1]
                emb_lyr = nn.Embedding(n, r)
                emb_lyr.weight.data.copy_(torch.FloatTensor(emb))
            else:
                r = n_components
                emb_lyr = nn.Embedding(n, r)
            emb_lyr.weight.requires_grad = train
            self.embs[k] = nn.Sequential(
                emb_lyr,
                nn.Linear(r, n_hid), self.non_lin(),
                nn.Linear(n_hid, n_hid), self.non_lin()
            )
            self.add_module(k, self.embs[k])

        self.user_rnn = nn.LSTM(n_hid, n_hid, n_layers, batch_first=True)
        self.item_rnn = nn.LSTM(n_hid, n_hid, n_layers, batch_first=True)

        if learn_metric:
            self.metric = nn.Sequential(
                nn.Linear(n_hid * 2, n_out),
                self.non_lin(),
                nn.Linear(n_hid, 1)
            )
            self.add_module('metric', self.metric)
            # self.metric = nn.Linear(n_hid * 2, 1)
        else:
            self.user_out = nn.Linear(n_hid, n_out)
            self.item_out = nn.Linear(n_hid, n_out)

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
        hid_u = pid.unsort(hid_u[0][-1])  # only take last rnn layer
        hid_i = tid.unsort(hid_i[0][-1])
        # hid_u = hid_u.permute(1, 0, 2).contiguous()
        # hid_u = hid_u.view(hid_u.shape[0], -1)
        # hid_i = hid_i.permute(1, 0, 2).contiguous()
        # hid_i = hid_i.view(hid_i.shape[0], -1)
        # hid_u = pid.unsort(hid_u)
        # hid_i = pid.unsort(hid_i)

        # obtain final estimation
        if self.learn_metric:
            # concat state
            h = torch.cat([hid_u, hid_i], dim=-1)  # (batch, n_hid * 2)
            y_pred = self.metric(h)[:, 0]
        else:
            y_pred = torch.bmm(
                self.user_out(hid_u).view(hid_u.shape[0], 1, self.n_out),
                self.item_out(hid_i).view(hid_i.shape[0], self.n_out, 1)
            ).squeeze()
            # y_pred = torch.sum(hid_u * hid_i, dim=-1)

        return y_pred

    def user_factor(self, pid):
        """"""
        emb_pl = self.embs['user'](Variable(pid.seq))
        emb_pl = pack_padded_sequence(emb_pl, pid.lengths.tolist(), batch_first=True)
        out_u, hid_u = self.user_rnn(emb_pl)
        return self.user_out(pid.unsort(hid_u[0][-1]))

    def item_factor(self, tid):
        """"""
        emb_tr = self.embs['item'](Variable(tid.seq))
        emb_tr = pack_padded_sequence(emb_tr, tid.lengths.tolist(), batch_first=True)
        out_i, hid_i = self.item_rnn(emb_tr)
        return self.item_out(tid.unsort(hid_i[0][-1]))


if __name__ == "__main__":
    """"""
    # setup some aliases
    HP = CONFIG['hyper_parameters']

    # make ngram dicts
    print('Preparing {:d}-Gram dictionary...'.format(HP['ngram_n']))
    uniq_playlists = pd.read_csv(CONFIG['path']['data']['playlists'], sep='\t',
                                 index_col=None, header=None)
    uniq_ngrams_pl = get_unique_ngrams(uniq_playlists[1].values, n=HP['ngram_n'])
    playlist_dict_ = dict(uniq_playlists[[3, 1]].values)
    pl_ngram_dict = {v:k for k, v in enumerate(uniq_ngrams_pl)}

    playlist_dict = dict(zip(
        playlist_dict_.keys(),
        transform_id2ngram_id(
            playlist_dict_.keys(), playlist_dict_, pl_ngram_dict)
    ))

    print('Uniq {:d}-Gram for playlists: {:d}'
          .format(HP['ngram_n'], len(uniq_ngrams_pl)))

    # for track, we will use word2vec pretrained embedding
    # uniq_tracks = pd.read_csv(CONFIG['path']['data']['tracks'], sep='\t',
    #                           index_col=None, header=None)
    with open(CONFIG['path']['data']['tracks']) as f:
        uniq_tracks = pd.DataFrame(
            [ll.replace('\n', '').split('\t') for ll in f.readlines()]
        )
        uniq_tracks.loc[:, 0] = uniq_tracks[0].astype(int)
        uniq_tracks.loc[:, 3] = uniq_tracks[3].astype(int)
    if HP['track_emb'] == 'word2vec':
        nist = NISTTokenizer()
        titles = uniq_tracks[1].apply(
            lambda a: nist.international_tokenize(
                re.sub(r'\([^)]*\)', '', a).decode('utf-8'),
                lowercase=True
            )
        )
        # simple remedy
        missing = titles.apply(len) == 0
        titles[missing] = uniq_tracks[missing][1].apply(lambda x: x.lower().split())
        id2word = pkl.load(open(CONFIG['path']['embeddings']['track_id2word']))
        word2id = {v: k for k, v in enumerate(id2word)}
        track_dict = {
            k: [word2id[vv] for vv in v] for k, v in titles.items()}

        track_emb = np.load(CONFIG['path']['embeddings']['track_word'])
        n_words = len(id2word)
        train_emb = False
        print('Uniq Token for tracks: {:d}'.format(len(id2word)))
        print('# of 0 length seq: {:d}'.format(
            len(filter(lambda r: r == 0, map(len, track_dict.values())))
        ))

    elif HP['track_emb'] == 'ngram':
        uniq_ngrams_tr = get_unique_ngrams(uniq_tracks[1].values, n=HP['ngram_n'])
        track_dict_ = dict(uniq_tracks[[3, 1]].values)
        tr_ngram_dict = {v:k for k, v in enumerate(uniq_ngrams_tr)}

        track_dict = dict(zip(
            track_dict_.keys(),
            transform_id2ngram_id(
                track_dict_.keys(), track_dict_, tr_ngram_dict)
        ))

        track_emb = None
        n_words = len(uniq_ngrams_tr)
        train_emb = True
        print('Uniq {:d}-Gram for tracks: {:d}'
              .format(HP['ngram_n'], len(uniq_ngrams_tr)))

    # prepare model instances
    sampler = MPDSampler(CONFIG, verbose=True)
    model = CFRNN(
        n_components=HP['n_embedding'],
        n_hid=HP['n_hid'],
        n_out=HP['n_out_embedding'],
        n_layers=HP['n_layers'],
        non_lin=HP['non_lin'],
        n_users=len(uniq_ngrams_pl), n_items=n_words,
        user_emb=None, item_emb=track_emb, user_train=True, item_train=train_emb,
        learn_metric=HP['learn_metric']
    ).cuda()

    # set loss / optimizer
    f_loss = NEGCrossEntropyLoss().cuda()
    opt = HP['optimizer'](
        filter(lambda p: p.requires_grad, model.parameters()),
        weight_decay=HP['l2'], lr=HP['learn_rate'])

    # main training loop
    model.train()
    losses = []
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
                # pid = SeqTensor(pid, playlist_dict, pl_ngram_dict)
                # tid = SeqTensor(tid, track_dict, tr_ngram_dict)
                pid = SeqTensor([playlist_dict[i] for i in pid], None, None)
                tid = SeqTensor([track_dict[i] for i in tid], None, None)

                # flush grad
                opt.zero_grad()

                # forward pass
                y_pred = model.forward(pid, tid)

                # calc loss
                l = f_loss(y_pred, Variable(pref))

                # back-propagation
                l.backward()

                # # clip gradients
                # grad_norm = nn.utils.clip_grad_norm(
                #     filter(lambda p: p.requires_grad, model.parameters()),
                #     max_norm=100
                # )

                # update
                opt.step()

                # log
                # epoch.set_description(
                #     '[loss : {:.3f} / |grad|: {:.2f}]'.format(float(l.data), grad_norm)
                # )
                epoch.set_description(
                    '[loss : {:.3f}]'.format(float(l.data))
                )
                losses.append(float(l.data))

    except KeyboardInterrupt:
        print('[Warning] User stopped the training!')
    # switch off to evaluation mode
    model.eval()

    # 1) extract playlist / track factors first from embedding-rnn blocks
    b = 500
    m = uniq_tracks.shape[0]
    tid = range(m)
    #   1.1) ext item factors first
    Q = []
    for j in trange(0, m+(m%b), b, ncols=80):
        if j > m:
            continue
        tid_ = SeqTensor([track_dict[jj] for jj in tid[j:j+b]], None, None)
        Q.append(model.item_factor(tid_).data.cpu().numpy())
    Q = np.concatenate(Q, axis=0)
    np.save('/tudelft.net/staff-bulk/ewi/insy/MMC/jaykim/datasets/recsys18/title_rnn_V.npy', Q)

    #   1.2) ext playlist factors
    n = uniq_playlists.shape[0]
    pid = range(n)
    P = []
    for j in trange(0, n+(n%b), b, ncols=80):
        if j > n:
            continue
        pid_ = SeqTensor([playlist_dict[jj] for jj in pid[j:j+b]], None, None)
        P.append(model.user_factor(pid_).data.cpu().numpy())
    P = np.concatenate(P, axis=0)
    np.save('/tudelft.net/staff-bulk/ewi/insy/MMC/jaykim/datasets/recsys18/title_rnn_U.npy', P)

    if sampler.test is not None:
        print('Evaluate!')
        # fetch testing playlists
        trg_u = sampler.test['playlist'].unique()

        #   2) calculate scores using the same way of MLP case
        rprec = []
        ndcg = []
        for j, u in tqdm(enumerate(trg_u), total=len(trg_u), ncols=80):
            true = sampler.test[sampler.test['playlist'] == u]['track']
            true_t = set(sampler.train[sampler.train['playlist'] == u]['track'])

            # predict k
            if model.learn_metric:
                cat = np.concatenate([np.tile(P[u], (Q.shape[0], 1)), Q], axis=-1)
            pred = []
            for k in trange(0, m, b, ncols=80):
                if model.learn_metric:
                    pred.append(model.metric(Variable(torch.cuda.FloatTensor(cat[k:k+b])))[:, 0].data)
                else:
                    pred.append(P[u].dot(Q[k:k+b].T))
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

    # 3) print out the loss evolution as a image
    plt.plot(losses)
    plt.savefig('./data/losses.png')
