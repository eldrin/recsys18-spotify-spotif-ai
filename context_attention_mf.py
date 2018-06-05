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
            self.tr_emb = nn.Embedding(n_items, item_emb.shape[-1])
            self.tr_emb.weight.data.copy_(torch.FloatTensor(item_emb))
        else:
            self.tr_emb = nn.Embedding(n_items, self.n_out)
        self.tr_emb.weight.requires_grad = item_train

        # playlist embedding
        self.proj = nn.Linear(self.tr_emb.embedding_dim, n_hid)
        self.proj_t = nn.Linear(self.tr_emb.embedding_dim, self.n_out)
        self.att_w = nn.Linear(n_hid, n_hid)
        self.att_h = nn.Linear(n_hid, 1)

        self.pl_emb = nn.Linear(n_hid, self.n_out)

    def _user_out(self, contexts):
        """"""
        # get context embedding
        x = self.tr_emb(Variable(contexts.seq))  # (n_batch, n_steps, n_embs)
        x = swish(self.proj(x))

        # FC Attention
        z = swish(self.att_w(x))
        a = F.softmax(self.att_h(z).squeeze(), dim=1)[:, :, None]
        x = torch.sum(x * a, dim=1).squeeze()

        # OUT FC (playlist embedding)
        x = self.pl_emb(x)  # (n_batch, dim_emb)

        return x

    def _item_out(self, tid):
        """"""
        y = self.tr_emb(tid)  # (n_batch, dim_emb)
        y = self.proj_t(swish(y))
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
            x.view(x.shape[0], 1, self.n_out),
            y.view(y.shape[0], self.n_out, 1)
        ).squeeze()
        return pred


if __name__ == "__main__":
    """"""
    # setup some aliases
    HP = CONFIG['hyper_parameters']

    # load audio_feature
    X = np.load(CONFIG['path']['embeddings']['X'])

    # prepare model instances
    sampler = MPDSampler(CONFIG, verbose=True)
    model = ItemAttentionCF(
        n_components=HP['n_embedding'],
        n_hid=HP['n_hid'],
        n_out=HP['n_out_embedding'],
        non_lin=HP['non_lin'],
        n_users=sampler.n_playlists, n_items=sampler.n_tracks,
        user_emb=None, item_emb=X, user_train=True, item_train=False,
        learn_metric=HP['learn_metric']
    ).cuda()

    mf = CFMLP(
        n_components=HP['n_out_embedding'], architecture=[],
        n_users=sampler.n_playlists, n_items=sampler.n_tracks,
        user_train=True, item_train=True,
        user_emb=None, item_emb=None, learn_metric=False).cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    params += filter(lambda p: p.requires_grad, mf.parameters())

    # set loss / optimizer
    f_loss = NEGCrossEntropyLoss().cuda()
    opt = HP['optimizer'](params, weight_decay=HP['l2'], lr=HP['learn_rate'])

    # main training loop
    model.train()
    mf.train()
    try:
        epoch = trange(HP['num_epochs'], ncols=80)
        for n in epoch:
            for batch, context in sampler.generator():
                batch_t = np.array(batch).T
                C = SeqTensor(map(list, context))

                # parse in / out
                pid, tid = batch_t[:2]
                pid = torch.cuda.LongTensor(pid)
                tid = torch.cuda.LongTensor(tid)
                pref, conf = [
                    torch.cuda.FloatTensor(a) for a in batch_t[-2:]]

                # flush grad
                opt.zero_grad()

                # forward pass
                y_pred = model.forward(pid, tid, C)
                y_pred += mf.forward(pid, tid)

                # calc loss
                l = f_loss(y_pred, Variable(pref))

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
    mf.eval()

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

        tid_ = torch.cuda.LongTensor(range(i, m_))
        Q.append(
            torch.cat([
                mf.mlps['item'](tid_), model._item_out(tid_)
            ], dim=1).data.cpu().numpy()
        )
    Q = np.concatenate(Q, axis=0)
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
                plst.append(np.random.choice(10000, 10).tolist())
        C = SeqTensor(plst)
        P.append(
            torch.cat([
                mf.mlps['user'](torch.cuda.LongTensor(m[slice(i, i+b)])),
                model._user_out(C)], dim=1).data.cpu().numpy())
    P = np.concatenate(P, axis=0)
    print(P.shape)
    # np.save('./data/cnn_U.npy', Q)

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
            pred = []
            for k in trange(0, len(m), b, ncols=80):
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
