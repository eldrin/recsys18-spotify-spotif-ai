import os
from functools import partial

import numpy as np
import pandas as pd

from model import ImplicitALS
from tqdm import tqdm
import fire

import sys
sys.path.append(os.path.join(os.getcwd(), 'RecsysChallengeTools'))
from metrics import ndcg, r_precision, playlist_extender_clicks
from util import sigmoid, BaseMF, MultiMF

# GLOBAL SETTINGS
CUTOFF = 500
NDCG = partial(ndcg, k=CUTOFF)


def _evaluate_playlist(true, pred):
    """"""
    return {
        'NDCG': NDCG(list(true), pred),
        'R_Precision': r_precision(list(true), pred),
        'Playlist_Extender_Clicks': playlist_extender_clicks(list(true), pred)
    }


def _evaluate_all(trues, preds, reduce='mean'):
    """"""
    res = map(_evaluate_playlist, trues, preds)
    df = pd.DataFrame(res)
    return df.mean()


def evaluate(model, y, yt, cutoff=CUTOFF):
    """"""
    trg_pl = yt['playlist'].unique()
    trues, preds = [], []
    yt_tracks = yt.groupby('playlist')['track'].apply(set)
    y_tracks = y[y['value']==1].groupby('playlist')['track'].apply(set)
    for pid in tqdm(trg_pl, total=len(trg_pl), ncols=80):
        true_ts = yt_tracks.loc[pid]
        if pid in y_tracks.index:
            true_tr = y_tracks.loc[pid]
        else:
            true_tr = set()
        pred = model.predict_k(pid, k=cutoff * 2).ravel()

        # exclude training data
        pred = filter(lambda x: x not in true_tr, pred)[:cutoff]

        trues.append(true_ts)
        preds.append(pred)

    # calc measures
    res = _evaluate_all(trues, preds)
    return res, (trues, preds)


def evaluate_mf(
    user_factor_fn, item_factor_fn, train_fn, test_fn, cutoff=CUTOFF,
    mode='all', **kwargs):
    """
    Args:

        user_factor_fn (str) : path to user factor
        item_factor_fn (str) : path to item factor
        train_fn (str) : path to train data
        test_fn (str) : path to test data
        cutoff (int) : cutoff for calcultate top-k metrics (default : 500)
        mode (str) : evaluation mode {'all', 'no_seed', 'only_seed'}
            'all': evaluate the mf model for all test case (default)
            'no_seed': evaluate the mf model only for the **no-seed** test cases
            'only_seed': evaluate the mf model only for the **seed** test cases
        **kwargs : mf arguments (i.e. importance, logistic...)
    """
    print('Load data...')
    y = pd.read_csv(train_fn, header=None, names=['playlist', 'track', 'value'])
    yt = pd.read_csv(test_fn, header=None, names=['playlist', 'track', 'value'])

    if mode == 'no_seed':
        yt = yt[yt['playlist'].isin(set(y[y['value']==0]['playlist'].unique()))]
    elif mode == 'only_seed':
        yt = yt[~yt['playlist'].isin(set(y[y['value']==0]['playlist'].unique()))]

    print('Load factors and initiate model...')
    # setup model
    mf = BaseMF(np.load(user_factor_fn), np.load(item_factor_fn), **kwargs)
    model = MultiMF(mf)

    print('Evaluate...')
    res, (trues, preds) = evaluate(model, y, yt, cutoff=CUTOFF)
    print res


if __name__ == "__main__":
    fire.Fire(evaluate_mf)
