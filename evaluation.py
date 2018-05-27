import os
from functools import partial

import numpy as np
import pandas as pd

from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.getcwd(), 'RecsysChallengeTools'))
from metrics import ndcg, r_precision, playlist_extender_clicks

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
        pred = model.predict_k(pid, k=cutoff * 2)

        # exclude training data
        pred = filter(lambda x: x not in true_tr, pred)[:cutoff]

        trues.append(true_ts)
        preds.append(pred)

    # calc measures
    res = _evaluate_all(trues, preds)
    return res
