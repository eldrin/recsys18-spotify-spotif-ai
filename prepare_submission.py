import os
import cPickle as pkl
import json

import pandas as pd
import numpy as np

from util import read_hash

from tqdm import trange, tqdm
import fire


def sigmoid(x):
    """"""
    return 1./(1. + np.exp(-x))

class BaseMF:
    """"""
    def __init__(self, P, Q, importance=1, logistic=False, name='MF', **kwargs):
        """"""
        self.P = P
        self.Q = Q
        self.logistic = logistic
        self.a = importance
        self.name = name

    def predict_score(self, u):
        """"""
        score = self.P[u].dot(self.Q.T)
        return sigmoid(score) if self.logistic else score


class MultiMF:
    """"""
    def __init__(self, *mfmodels):
        """"""
        self.models = list(mfmodels)

    def predict_k(self, pids, k=500):
        """"""
        # get scores
        scores = -np.sum(
            [mf.a * mf.predict_score(pids) for mf in self.models], axis=0
        )
        ix = np.argpartition(scores, k, axis=1)[:, :k]
        pred_raw_ix = scores[np.arange(ix.shape[0])[:,None], ix].argsort(1)
        pred_raw_batch = ix[np.arange(ix.shape[0])[:,None], pred_raw_ix]
        return pred_raw_batch


def main(config_path, n_rec=500, batch_size=100):
    """"""
    # load configuration file
    config = json.load(open(config_path))
    model_paths = config['path']['models']
    data_paths = config['path']['data']
    out_fn = config['path']['output']

    # load hashes (fac_id >> org_id)
    pl_hash = {int(k): v for k, v in read_hash(data_paths['playlists'])[[3, 2]].values}
    tr_hash = {int(k): v for k, v in read_hash(data_paths['tracks'])[[3, 2]].values}

    # get pl inv hash (org_pid >> fac_id)
    pid2ix = {int(v): k for k, v in pl_hash.iteritems()}

    # setup model
    models = [
        BaseMF(
            np.load(args['P']), np.load(args['Q']),
            importance=args['importance'],
            logistic=args['logistic']
        )
        for name, args in model_paths.iteritems()
        if args['name'] != 'na'
    ]
    model = MultiMF(*models)

    # get candidates
    queries = pd.DataFrame(
        json.load(open(data_paths['challenge_set']))['playlists']
    )

    # predict!
    sanity = {}
    predictions = {}
    query_tracks = {}
    n_cand = int(n_rec * 1.2)
    for ix in trange(0, queries.shape[0], batch_size, ncols=80):
        qry_slc = queries.iloc[slice(ix, ix+batch_size)]
        pids = qry_slc['pid']
        names = qry_slc['name']
        tracks = qry_slc['tracks']

        # get initial prediction (for batch, 1k candidates)
        pids_ = [pid2ix[ix] for ix in pids.values]
        pred_raw_batch = model.predict_k(pids_, k=n_cand)

        # quick sanity check
        assert tracks.shape[0] == pred_raw_batch.shape[0]

        # start process
        for pid, pred_raw, tracks_ in zip(pids.values, pred_raw_batch, tracks.values):
            tracks_set = set([x['track_uri'] for x in tracks_])

            # unhashing (fac_id >> uri) (filter out queries)
            preds = [tr_hash[p] for p in pred_raw]
            pred = filter(lambda tr: tr not in tracks_set, preds)[:n_rec]
            predictions[pid] = pred
            query_tracks[pid] = tracks_set

            # sanity check
            sanity[pid] = bool(len(pred) < n_cand)
    print('Sanity: {:d} / {:d}'.format(sum(sanity.values()), len(sanity)))

    # writing submission file
    pkl.dump(query_tracks,
             open(os.path.join(os.path.dirname(out_fn), 'queries.pkl'), 'wb'))
    with open(out_fn, 'w') as f:
        f.write("team_info,creative,spotif.ai,J.H.Kim@tudelft.nl\n")
        for pid, track_uris in tqdm(predictions.items(), ncols=80):
            f.write("{:d},".format(pid) + ",".join(track_uris) + "\n")

    # TODO: add verifying process using `verify_submission.py`
    #       and gzip it in the script


if __name__ == "__main__":
    fire.Fire(main)
