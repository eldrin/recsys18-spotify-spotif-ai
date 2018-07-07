import os
from functools import partial
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable

import sys
sys.path.append(os.path.join(os.getcwd(), 'RecsysChallengeTools/'))
from metrics import ndcg, r_precision, playlist_extender_clicks
NDCG = partial(ndcg, k=500)

from data import MPDSampler
from data import get_unique_ngrams
from data import transform_id2ngram_id
from model import UserRNN
from losses import SGNS, SGNSMSE
from optimizers import MultipleOptimizer
from util import read_hash

from tqdm import tqdm, trange
import fire


EARLY_STOP_K = 20000


def main(config_fn):
    """"""
    CONFIG = json.load(open(config_fn))

    # setup some aliases
    HP = CONFIG['hyper_parameters']
    USE_GPU = CONFIG['hyper_parameters']['use_gpu']
    EARLY_STOP = HP['early_stop']

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
        user_train=True, n_layers=HP['n_layers'],
        drop_out=HP['drop_out'], sparse_embedding=True
    )
    """"""
    track_factors = nn.Embedding(
        sampler.n_tracks, HP['n_out_embedding'], sparse=True)
    """"""

    # set loss / optimizer
    if HP['loss'] == 'MSE':
        # f_loss = nn.SmoothL1Loss().cuda()
        f_loss = nn.MSELoss()

        # load target (pre-trained) playlist factors
        playlist_factors = np.load(
            CONFIG['path']['embeddings']['U']).astype(np.float32)

    elif HP['loss'] == 'SGNS':
        f_loss = SGNS()

        # load target (pre-trained) playlist factors
        """
        track_factors = np.load(
            CONFIG['path']['embeddings']['V']).astype(np.float32)
        """

    elif HP['loss'] == 'all':
        # load target (pre-trained) playlist factors
        playlist_factors = np.load(
            CONFIG['path']['embeddings']['U']).astype(np.float32)

        # load target (pre-trained) playlist factors
        track_factors = np.load(
            CONFIG['path']['embeddings']['V']).astype(np.float32)

        # setup total loss
        f_loss = SGNSMSE()

    sprs_prms = map(
        lambda x: x[1],
        filter(lambda kv: kv[0] in {'emb.weight'},
               model.named_parameters())
    )
    """"""
    if HP['loss'] == 'SGNS':
        sprs_prms.append(track_factors.weight)
    """"""
    dnse_prms = map(
        lambda x: x[1],
        filter(lambda kv: kv[0] not in {'emb.weight'},
               model.named_parameters())
    )
    opt = MultipleOptimizer(
        optim.SparseAdam(sprs_prms, lr=HP['learn_rate']),
        # optim.Adam(dnse_prms, lr=HP['learn_rate'], amsgrad=True)  # <= 0.4.0 pytorch
        optim.Adam(dnse_prms, lr=HP['learn_rate'])  # > 0.4.0 pytorch
    )

    if USE_GPU:
        model = model.cuda()
        model.is_cuda = True
        f_loss = f_loss.cuda()

    # main training loop
    model.train()
    losses = []
    try:
        epoch = trange(HP['num_epochs'], ncols=80)
        stop = False
        k = 0
        for n in epoch:
            if stop:
                break
            for batch in sampler.generator():
                k += 1
                # arbitrary early stop used for fast training
                # NOTE: only works with 'all' loss
                if EARLY_STOP:
                    if k > EARLY_STOP_K:
                        stop = True
                        break

                # parse in / out
                pid_ = [x[0] for x in batch]  # (n_batch,)
                pid = [playlist_dict[i] for i in pid_]
                conf_ = [x[-1] for x in batch]
                conf = Variable(torch.FloatTensor(conf_))

                # flush grad
                opt.zero_grad()

                # forward pass
                y_pred = model.forward(pid)

                if HP['loss'] == 'MSE':
                    y_true = Variable(
                        torch.from_numpy(playlist_factors[pid_])
                    )
                    if USE_GPU:
                        y_true = y_true.cuda()

                    # calc loss
                    l = f_loss(y_pred, y_true)

                elif HP['loss'] == 'SGNS':
                    tid_ = [[x[1]] + x[2] for x in batch]
                    pref_ = [[1.] + [-1.] * len(x[2]) for x in batch]
                    pref = Variable(torch.FloatTensor(pref_))

                    # calc loss
                    """
                    v = Variable(
                        torch.from_numpy(
                            np.array([track_factors[t] for t in tid_])
                        )
                    )
                    """
                    """"""
                    v = track_factors(
                        Variable(torch.LongTensor(tid_))
                    )
                    """"""

                    if USE_GPU:
                        pref = pref.cuda()
                        v = v.cuda()

                    hv = torch.bmm(
                        v, y_pred.view(y_pred.shape[0], y_pred.shape[-1], 1)
                    ).squeeze()
                    l = f_loss(hv, pref)
                    l += HP['l2'] * ((y_pred)**2).mean()**.5
                    l += HP['l2'] * (v**2).mean()**.5

                elif HP['loss'] == 'all':
                    tid_ = [[x[1]] + x[2] for x in batch]
                    pref_ = [[1.] + [-1.] * len(x[2]) for x in batch]
                    pref = Variable(torch.FloatTensor(pref_))
                    y_true = Variable(torch.from_numpy(playlist_factors[pid_]))
                    v = Variable(
                        torch.from_numpy(
                            np.array([track_factors[t] for t in tid_])
                        )
                    )

                    if USE_GPU:
                        pref = pref.cuda()
                        y_true = y_true.cuda()
                        v = v.cuda()

                    # calc loss
                    l = f_loss(y_pred, v, y_true, pref)

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
    """"""
    if HP['loss'] == 'SGNS':
        Q = track_factors.weight.data.numpy()
    else:
        Q = track_factors
    # np.save(CONFIG['path']['model_out']['V'], Q)
    # del track_factors, Q
    # Q = np.load(
    #     CONFIG['path']['model_out']['V'], mmap_mode='r')
    """"""

    #   1.2) ext playlist factors
    b = 500
    n = uniq_playlists.shape[0]
    pid = range(n)
    P = []
    for j in trange(0, n+(n%b), b, ncols=80):
        if j > n:
            continue
        pid_ = [playlist_dict[jj] for jj in pid[j:j+b]]
        if USE_GPU:
            P.append(model.user_factor(pid_).data.cpu().numpy())
        else:
            P.append(model.user_factor(pid_).data.numpy())

    P = np.concatenate(P, axis=0).astype(np.float32)
    np.save(CONFIG['path']['model_out']['U'], P)
    del P
    P = np.load(
        CONFIG['path']['model_out']['U'], mmap_mode='r')

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

    # 4) dump the rnn model / ngram dict
    checkpoint = {
        'epoch': n, 'updates': k,
        'state_dict': model.state_dict(),
        # 'optimizer': opt.state_dict(),  # TODO
        'uniq_ngram': uniq_ngrams_pl
    }
    torch.save(checkpoint, CONFIG['path']['model_out']['rnn'])


if __name__ == "__main__":
    fire.Fire(main)
