import os
import re
from functools import partial
from collections import namedtuple
import cPickle as pkl
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
sys.path.append(os.path.join(os.getcwd(), 'RecsysChallengeTools'))
from metrics import ndcg, r_precision, playlist_extender_clicks
NDCG = partial(ndcg, k=500)

from mfmlp import MPDSampler, NEGCrossEntropyLoss
from cfatt_conf import CONFIG
from context_attention_mf import MF

if __name__ == "__main__":
    """"""
    # setup some aliases
    HP = CONFIG['hyper_parameters']

    # prepare model instances
    sampler = MPDSampler(CONFIG, verbose=True)
    mf = MF(
        n_components=HP['n_embedding'],
        n_users=sampler.n_playlists,
        n_items=sampler.n_tracks,
        user_emb=None, item_emb=None,
        user_train=True, item_train=True,
        sparse_embedding=True
    ).cuda()

    params = filter(lambda p: p.requires_grad, mf.parameters())

    # set loss / optimizer
    f_loss = NEGCrossEntropyLoss().cuda()
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
                y_pred = mf.forward(pid, tid)

                # calc loss
                l = f_loss(y_pred, pref, conf)
                # l = f_loss(y_pred, pref)

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

    # 1) get item factors first
    Q = mf.item_emb.weight.data.cpu().numpy()

    # 2) get playlist factors
    P = mf.user_emb.weight.data.cpu().numpy()

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
            pred = []
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
