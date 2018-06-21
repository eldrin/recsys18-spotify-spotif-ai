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
from context_attention_mf import MF

ON_GPU = True
IS_PCA = False

if __name__ == "__main__":
    """"""
    # setup some aliases
    HP = CONFIG['hyper_parameters']

    # load item embedding, if available
    x_fn = CONFIG['path']['embeddings']['X']
    if isfile(x_fn):
        item_factors = np.load(x_fn)
        item_train = False
        n_components = item_factors.shape[-1]
        if IS_PCA:
            pca = PCA(whiten=True)
            item_factors = pca.fit_transform(item_factors)

    else:
        item_factors = None
        item_train = True
        n_components = HP['n_embedding']


    # prepare model instances
    sampler = MPDSampler(CONFIG, verbose=True)
    mf = MF(
        n_components=n_components,
        n_users=sampler.n_playlists,
        n_items=sampler.n_tracks,
        user_emb=None, item_emb=item_factors,
        user_train=True, item_train=item_train,
        sparse_embedding=True
    )
    if ON_GPU:
        mf = mf.cuda()

    params = filter(lambda p: p.requires_grad, mf.parameters())

    # set loss / optimizer
    f_loss = NEGCrossEntropyLoss()
    if ON_GPU:
        f_loss = f_loss.cuda()
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
                y_pred, p_, q_ = mf.forward(pid, tid)

                # calc loss
                l = f_loss(y_pred, pref, conf)
                # l = f_loss(y_pred, pref)
                # l += HP['l2'] * (p_**2).mean()**.5

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
        P = mf.user_emb.weight.data.cpu().numpy()
        Q = mf.item_emb.weight.data.cpu().numpy()
    else:
        Q = mf.item_emb.weight.data.numpy()
        P = mf.user_emb.weight.data.numpy()

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

    # save!
    np.save('./data/afeat_spotify_U.npy', P)
