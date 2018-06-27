import os
from os.path import isfile
import re
from functools import partial
from collections import namedtuple
import cPickle as pkl
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from prefetch_generator import background
from tqdm import tqdm, trange

# from evaluation import r_precision, NDCG
import sys
sys.path.append(os.path.join(os.getcwd(), 'RecsysChallengeTools'))
from metrics import ndcg, r_precision, playlist_extender_clicks
NDCG = partial(ndcg, k=500)

from mfmlp import MPDSampler, NEGCrossEntropyLoss
from simplemf_conf import CONFIG
from context_attention_mf import MF, init_embedding_layer, check_emb_shape
from util import MultipleOptimizer

ON_GPU = True
IS_PCA = False


def load_embedding(emb_fn, d, is_pca):
    if isfile(emb_fn):
        item_factors = np.load(emb_fn)
        item_train = False
        n_components = item_factors.shape[-1]
        if is_pca:
            pca = PCA(whiten=True)
            item_factors = pca.fit_transform(item_factors)
    else:
        item_factors = None
        item_train = True
        n_components = d

    return item_factors, item_train, n_components


class NNMF(nn.Module):
    """"""
    def __init__(self, n_components, n_hid, n_users, n_items, user_emb, item_emb,
                 user_train=True, item_train=True, sparse_embedding=True):
        """"""
        super(NNMF, self).__init__()
        self.n_components = n_components
        self.n_users = n_users
        self.n_items = n_items

        # check_emb_shape(user_emb, item_emb, n_components)

        # user embedding
        user_r, self.user_emb = init_embedding_layer(
            user_emb, n_users, n_components, user_train, sparse_embedding)

        # item embedding
        # item_r, self.item_emb = init_embedding_layer(
        #     item_emb, n_items, n_components, item_train, sparse_embedding)
        item_r, self.item_emb = init_embedding_layer(
            item_emb, n_items, n_hid, item_train, sparse_embedding)

        # # metric layers
        # self.h1_user = nn.Linear(user_r, n_hid)
        # self.h1_item = nn.Linear(item_r, n_hid)
        # self.h2 = nn.Linear(n_hid, n_hid)
        # self.h3 = nn.Linear(n_hid, 1)

        self.hu1 = nn.Linear(user_r, n_hid)
        # self.hu2 = nn.Linear(n_hid, n_hid)
        # self.hu3 = nn.Linear(n_hid, n_hid)
        # self.hu4 = nn.Linear(n_hid, n_hid)
        # self.hu5 = nn.Linear(n_hid, n_hid)
        self.huo = nn.Linear(n_hid, n_hid)

        # self.hi1 = nn.Linear(item_r, n_hid)
        # self.hio = nn.Linear(n_hid, n_hid)

    # def _metric(self, p, q):
    #     """"""
    #     h = F.relu(self.h1_user(p) + self.h1_item(q))
    #     h = F.relu(self.h1(h))
    #     h = F.relu(self.h2(h))
    #     return self.h3(h).squeeze()

    def user_factor(self, p):
        """"""
        p = F.relu(self.hu1(p))
        # p = F.selu(self.hu2(p))
        # p = F.selu(self.hu3(p))
        # p = F.selu(self.hu4(p))
        # p = F.selu(self.hu5(p))
        p = self.huo(p)
        return p

    def item_factor(self, q):
        """"""
        # q = F.relu(self.hi1(q))
        # q = self.hio(q)
        return q

    def forward(self, u, i):
        """"""
        p = self.user_emb(u)
        q = self.item_emb(i)
        # pred = self._metric(p, q)
        p = self.user_factor(p)
        q = self.item_factor(q)

        if p.ndimension() == 2:
            p = p.view(p.shape[0], 1, p.shape[-1])

        if q.ndimension() == 2:
            q = q.view(q.shape[0], q.shape[-1], 1)
        elif q.ndimension() == 3:
            q = q.permute(0, 2, 1)  # (n_batch, n_hid, n_neg)

        pred = torch.bmm(p, q).squeeze()
        return pred, p, q


if __name__ == "__main__":
    """"""
    # setup some aliases
    HP = CONFIG['hyper_parameters']

    # load item embedding, if available
    x_fn = CONFIG['path']['embeddings']['X']
    item_factors, item_train, n_components = load_embedding(
        x_fn, HP['n_embedding'], IS_PCA)
    # item_factors, item_train, n_components = None, True, HP['n_hid']

    # u_fn = CONFIG['path']['embeddings']['U']
    # playlist_factors, playlist_train, n_components = load_embedding(
    #     u_fn, HP['n_embedding'], IS_PCA)

    # prepare model instances
    print('Init sampler!')
    sampler = MPDSampler(CONFIG, verbose=True)

    print('Init model!')
    mf = MF(
        n_components=n_components,
        n_users=sampler.n_playlists,
        n_items=sampler.n_tracks,
        user_emb=None, item_emb=item_factors,
        user_train=True, item_train=item_train,
        sparse_embedding=True
    )
    mf2 = MF(
        n_components=n_components,
        n_users=sampler.n_playlists,
        n_items=sampler.n_tracks,
        user_emb=None, item_emb=None,
        user_train=True, item_train=True,
        sparse_embedding=True
    )
    # mf = NNMF(
    #     n_components=n_components,
    #     n_hid=300,
    #     n_users=sampler.n_playlists,
    #     n_items=sampler.n_tracks,
    #     user_emb=playlist_factors, item_emb=item_factors,
    #     user_train=playlist_train, item_train=item_train,
    #     sparse_embedding=True
    # )
    if ON_GPU:
        mf = mf.cuda()
        mf2 = mf.cuda()

    # set loss / optimizer
    f_loss = NEGCrossEntropyLoss()
    if ON_GPU:
        f_loss = f_loss.cuda()

    # sprs_prms = map(
    #     lambda x: x[1],
    #     filter(lambda kv: 'emb' in kv[0] and kv[1].requires_grad,
    #            mf.named_parameters())
    # )
    # dnse_prms = map(
    #     lambda x: x[1],
    #     filter(lambda kv: 'emb' not in kv[0],
    #            mf.named_parameters())
    # )
    # opt = MultipleOptimizer(
    #     optim.SparseAdam(sprs_prms, lr=HP['learn_rate']),
    #     # optim.Adam(dnse_prms, lr=HP['learn_rate'], amsgrad=True)
    #     optim.Adam(dnse_prms, lr=HP['learn_rate'])
    # )

    params = filter(lambda p: p.requires_grad, mf.parameters())
    # opt = HP['optimizer'](params, weight_decay=HP['l2'], lr=HP['learn_rate'])
    opt = HP['optimizer'](params, lr=HP['learn_rate'])

    # main training loop
    mf.train()
    try:
        epoch = trange(HP['num_epochs'], ncols=80)
        for n in epoch:
            for batch in sampler.generator():

                # parse in / out
                tid_ = [[x[1]] + x[2] for x in batch]
                pid_ = [x[0] for x in batch]
                pref_ = [[1.] + [-1.] * len(x[2]) for x in batch]
                conf_ = [x[-1] for x in batch]

                if ON_GPU:
                    pid = Variable(torch.LongTensor(pid_).cuda())
                    tid = Variable(torch.LongTensor(tid_).cuda())
                    pref = Variable(torch.FloatTensor(pref_).cuda())
                    conf = Variable(torch.FloatTensor(conf_).cuda())
                else:
                    pid = Variable(torch.LongTensor(pid_))
                    tid = Variable(torch.LongTensor(tid_))
                    pref = Variable(torch.FloatTensor(pref_))
                    conf = Variable(torch.FloatTensor(conf_))

                # flush grad
                opt.zero_grad()

                # forward pass
                pq, p_, q_ = mf.forward(pid, tid)
                pq2, p_2, q_2 = mf2.forward(pid, tid)

                # calc loss
                # l = f_loss(y_pred, pref, conf)
                l = f_loss(pq + pq2, pref)
                l += HP['l2'] * ((p_+p_2)**2).mean()**.5
                l += HP['l2'] * ((q_+q_2)**2).mean()**.5

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
    mf.eval()

    # get factors
    if ON_GPU:
        # P = mf.user_emb.weight.data.cpu().numpy()
        # Q = mf.item_emb.weight.data.cpu().numpy()
        P = np.c_[
            mf.user_emb.weight.data.cpu().numpy(),
            mf2.user_emb.weight.data.cpu().numpy()
        ]
        Q = np.c_[
            mf.item_emb.weight.data.cpu().numpy(),
            mf2.item_emb.weight.data.cpu().numpy()
        ]

        # P = mf.user_factor(mf.user_emb.weight).data.cpu().numpy()
        # Q = mf.item_factor(mf.item_emb.weight).data.cpu().numpy()
    else:
        # Q = mf.item_emb.weight.data.numpy()
        # P = mf.user_emb.weight.data.numpy()
        P = np.c_[
            mf.user_emb.weight.data.numpy(),
            mf2.user_emb.weight.data.numpy()
        ]
        Q = np.c_[
            mf.item_emb.weight.data.cpu().numpy(),
            mf2.item_emb.weight.data.cpu().numpy()
        ]
        # P = mf.user_factor(mf.user_emb.weight).data.numpy()
        # Q = mf.item_factor(mf.item_emb.weight).data.numpy()

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

    # save!
    np.save('/mnt/bulk/recsys18/models/mf_w2v_U.npy', P)
    np.save('/mnt/bulk/recsys18/models/mf_w2v_V.npy', Q)
