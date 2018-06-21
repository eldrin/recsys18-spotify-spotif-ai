import os
import re
from functools import partial
from collections import namedtuple
import cPickle as pkl
import pandas as pd
import numpy as np
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
sys.path.append(os.path.join(os.getcwd(), 'RecsysChallengeTools'))
from metrics import ndcg, r_precision, playlist_extender_clicks
NDCG = partial(ndcg, k=500)

from mfmlp import MPDSampler, NEGCrossEntropyLoss, CFMLP
from data import get_ngram, get_unique_ngrams
from pretrain_word2vec import load_n_process_data
from rnn_nlp_cf import SeqTensor
from cfatt_conf import CONFIG

try:
    print(torch.cuda.current_device())
    floatX = torch.cuda.FloatTensor
except:
    floatX = torch.FloatTensor


def swish(x):
    """"""
    return x * x.sigmoid()


def mask_softmax(a, dim, epsilon=1e-8):
    """"""
    a_max = torch.max(a, dim=dim, keepdim=True)[0]
    a_exp = torch.exp(a - a_max)
    a_exp = a_exp * (a == 0).float()
    a_softmax = a_exp / (torch.sum(a_exp, dim=dim, keepdim=True) + epsilon)
    return a_softmax


def check_emb_shape(P, Q, r):
    """"""
    if P is not None and Q is not None:
        assert P.shape[1] == Q.shape[1]


def init_embedding_layer(emb, n, r, is_train, sparse=True):
    """"""
    if emb is not None:
        emb_layer = nn.Embedding(n, emb.shape[-1], sparse=sparse)
        emb_layer.weight.data.copy_(torch.FloatTensor(emb))

        if r != emb.shape[-1]:
            print('[Warning] n_components is not mathcing to input embedding\
                  overriding it to input embedding shape...')
            r = emb.shape[-1]
    else:
        emb_layer = nn.Embedding(n, r, sparse=sparse)
        emb_layer.weight.data.copy_(torch.randn(n, r) * 0.01)

    emb_layer.weight.requires_grad = is_train

    return r, emb_layer


class MF(nn.Module):
    """"""
    def __init__(self, n_components, n_users, n_items, user_emb, item_emb,
                 user_train=True, item_train=True, sparse_embedding=True):
        """"""
        super(MF, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_items = n_items

        check_emb_shape(user_emb, item_emb, n_components)

        # user embedding
        self.n_components, self.user_emb = init_embedding_layer(
            user_emb, n_users, n_components, user_train, sparse_embedding)

        # item embedding
        self.n_components, self.item_emb = init_embedding_layer(
            item_emb, n_items, n_components, item_train, sparse_embedding)

    def forward(self, u, i):
        """"""
        p = self.user_emb(u)
        q = self.item_emb(i)

        if p.ndimension() == 2:
            p = p.view(p.shape[0], 1, self.n_components)

        if q.ndimension() == 2:
            q = q.view(q.shape[0], self.n_components, 1)
        elif q.ndimension() == 3:
            q = q.permute(0, 2, 1)  # (n_batch, n_hid, n_neg)

        pred = torch.bmm(p, q).squeeze()
        return pred, p, q


class ItemAttentionCF(nn.Module):
    """"""
    def __init__(self, n_components, n_users, n_items,user_emb, item_emb,
                 n_hid=100, n_out=16, user_train=True, item_train=True,
                 non_lin=nn.ReLU, layer_norm=False, drop_out=0, learn_metric=True):
        """"""
        super(ItemAttentionCF, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_items = n_items
        self.non_lin = non_lin
        self.n_out = n_out
        self.learn_metric = learn_metric

        # track embedding
        if item_emb is not None:
            self.tr_emb = nn.Embedding(n_items, item_emb.shape[-1], sparse=True)
            self.tr_emb.weight.data.copy_(torch.FloatTensor(item_emb))
        else:
            self.tr_emb = nn.Embedding(n_items, self.n_out, sparse=True)
            self.tr_emb.weight.data.copy_(
                torch.randn(n_items, self.n_out) * 0.01)
        self.tr_emb.weight.requires_grad = item_train

        # track (out) embedding
        self.out_emb = nn.Embedding(n_items, self.n_out)
        self.out_emb.weight.requires_grad = True
        self.out_emb.weight.data.copy_(
            torch.randn(n_items, self.n_out) * 0.01)

        # playlist embedding
        self.proj = nn.Linear(self.tr_emb.embedding_dim, n_hid)
        self.att_w = nn.Linear(n_hid, n_hid)
        self.att_h = nn.Linear(n_hid, 1)
        self.att_bn = nn.BatchNorm1d(n_hid)

        self.pl_emb1 = nn.Linear(n_hid, self.n_out)
        # self.pl_emb2 = nn.Linear(n_hid, self.n_out)

    def _item_attention(self, contexts):
        """"""
        # get mask
        mask = (contexts != -1).float()  # (n_batch, n_seeds') ~ {0, 1}
        contexts[contexts == -1] = 0

        # get context embedding
        x = self.tr_emb(contexts)  # (n_batch, n_seeds', n_embs)
        x = x * mask[:, :, None]  # masking
        x = F.relu(self.proj(x))  # (n_batch, n_seeds', n_hid)

        # FC Attention
        z = F.relu(self.att_w(x))  # (n_batch, n_seeds', n_hid)
        a = mask_softmax(self.att_h(z).squeeze(), dim=1)

        return x, a

    def _user_out(self, contexts):
        """"""
        # get context embedding / attention
        x, a = self._item_attention(contexts)

        # get 1nd order momentum (mean)
        x1 = torch.bmm(a[:, None, :], x).squeeze()
        x1 = att_bn(x1)

        # # get 2nd order factor with attentional sum
        # sum_x_sqr = x1**2
        # sum_sqr_x = torch.bmm(a[:, None, :]**2, x**2).squeeze()
        # x2 = 0.5 * (sum_x_sqr - sum_sqr_x)  # covariance

        # # OUT FC (playlist embedding)
        # x = self.pl_emb1(x1) + self.pl_emb2(x2)  # (n_batch, dim_emb)

        x = self.pl_emb1(x1)

        return x

    def _item_out(self, tid):
        """"""
        y = self.out_emb(tid)  # (n_batch, dim_emb)
        return y

    def forward(self, pid, tid, contexts):
        """
        pid: SeqTensor instance for batch of playlist
        tid: SeqTensor instance for batch of tracks
        """
        # playlist emb
        x = self._user_out(contexts)

        # track emb
        y = self._item_out(tid)

        pred = torch.bmm(
            y, x.view(x.shape[0], self.n_out, 1)
        ).squeeze()
        return pred  # (n_bach, n_pos + n_neg)


if __name__ == "__main__":
    """"""
    # setup some aliases
    HP = CONFIG['hyper_parameters']

    # load audio_feature
    # X = np.load(CONFIG['path']['embeddings']['X'])
    # X = np.load(CONFIG['path']['embeddings']['V'])

    # prepare model instances
    sampler = MPDSampler(CONFIG, verbose=True)
    model = ItemAttentionCF(
        n_components=HP['n_embedding'],
        n_hid=HP['n_hid'],
        n_out=HP['n_out_embedding'],
        non_lin=HP['non_lin'],
        n_users=sampler.n_playlists, n_items=sampler.n_tracks,
        user_emb=None, item_emb=None, user_train=True, item_train=True,
        learn_metric=HP['learn_metric']
    ).cuda()

    # mf = MF(
    #     n_components=HP['n_embedding'],
    #     n_users=sampler.n_playlists,
    #     n_items=sampler.n_tracks,
    #     user_emb=None, item_emb=None,
    #     user_train=True, item_train=True
    # ).cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    # params += filter(lambda p: p.requires_grad, mf.parameters())

    # set loss / optimizer
    f_loss = NEGCrossEntropyLoss().cuda()
    # opt = HP['optimizer'](params, weight_decay=HP['l2'], lr=HP['learn_rate'])
    opt = HP['optimizer'](params, lr=HP['learn_rate'])

    # main training loop
    model.train()
    # mf.train()
    try:
        epoch = trange(HP['num_epochs'], ncols=80)
        for n in epoch:
            for batch in sampler.generator():

                # parse in / out
                tid_ = [[x[1]] + x[2] for x in batch]
                pid_ = [[x[0]] * len(tt) for x, tt in zip(batch, tid_)]
                pref_ = [[1.] + [-1.] * len(x[2]) for x in batch]
                conf_ = [x[-1] for x in batch]

                # get the mask
                context = [list(x[-1]) for x in batch]
                max_len = max(map(len, context))
                context = [a + [-1] * (max_len - len(a)) for a in context]

                pid = Variable(torch.LongTensor(pid_).cuda())
                tid = Variable(torch.LongTensor(tid_).cuda())
                pref = Variable(torch.FloatTensor(pref_).cuda())
                conf = Variable(torch.FloatTensor(conf_).cuda())
                context = Variable(torch.LongTensor(context).cuda())

                # flush grad
                opt.zero_grad()

                # forward pass
                y_pred = model.forward(None, tid, context)
                # y_pred += mf.forward(pid, tid)

                # calc loss
                l = f_loss(y_pred, pref, conf)

                # back-propagation
                l.backward()

                # update
                opt.step()

                epoch.set_description(
                    '[loss : {:.3f}]'.format(float(l.data))
                )

    except KeyboardInterrupt:
        print('[Warning] User stopped the training!')
    # switch off to evaluation mode
    model.eval()
    # mf.eval()

    b = 512

    # 1) get item factors first
    # P = model.tr_emb.weight.data.cpu().numpy()
    Q = []
    m = np.arange(sampler.n_tracks)
    for i in range(0, len(m), b):
        if i+b > len(m):
            m_ = len(m)
        else:
            m_ = i+b

        tid_ = Variable(torch.cuda.LongTensor(range(i, m_)))
        # Q.append(
        #     torch.cat([
        #         mf.mlps['item'](tid_), model._item_out(tid_)
        #     ], dim=1).data.cpu().numpy()
        # )
        Q.append(model._item_out(tid_).data.cpu().numpy())
    Q = np.concatenate(Q, axis=0).astype(np.float32)
    print(Q.shape)
    np.save('./data/cnn_V.npy', Q)

    # 2) get playlist factors
    P = []
    m = np.arange(sampler.n_playlists)
    for i in range(0, len(m), b):
        plst = []
        for pl in m[slice(i, i + b)]:
            if pl in sampler.pos_tracks:
                plst.append(list(sampler.pos_tracks[pl]))
            else:
                # pick popular items
                plst.append(np.random.choice(10000, 10).tolist())

        max_len = max(map(len, plst))
        context = [a + [-1] * (max_len - len(a)) for a in plst]
        C = Variable(torch.LongTensor(context).cuda())

        # P.append(
        #     torch.cat([
        #         mf.mlps['user'](Variable(torch.cuda.LongTensor(m[slice(i, i+b)]))),
        #         model._user_out(C)], dim=1).data.cpu().numpy())
        P.append(model._user_out(C).data.cpu().numpy())
    P = np.concatenate(P, axis=0).astype(np.float32)
    print(P.shape)
    # np.save('./data/cnn_U.npy', Q)

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
