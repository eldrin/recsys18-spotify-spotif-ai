import os
import cPickle as pkl
import json

import pandas as pd
import numpy as np

from util import read_hash

from tqdm import trange, tqdm
import fire

def main(out_fn, pl_fac_fn, tr_fac_fn, pl_hash_fn, tr_hash_fn, challenge_path, n_rec=500, batch_size=100):
    """"""
    # load hashes (fac_id >> org_id)
    pl_hash = {int(k): v for k, v in read_hash(pl_hash_fn)[[3, 2]].values}
    tr_hash = {int(k): v for k, v in read_hash(tr_hash_fn)[[3, 2]].values}

    # get pl inv hash (org_pid >> fac_id)
    pid2ix = {int(v): k for k, v in pl_hash.iteritems()}

    # load the factors
    U = np.load(pl_fac_fn)
    V = np.load(tr_fac_fn)

    # get candidates
    queries = pd.DataFrame(
        json.load(open(challenge_path))['playlists']
    )

    # predict!
    sanity = {}
    predictions = {}
    query_tracks = {}
    n_cand = int(n_rec * 1.5)
    for ix in trange(0, queries.shape[0], batch_size, ncols=80):
    # for ix in trange(2000, 3000, batch_size, ncols=80):
        qry_slc = queries.iloc[slice(ix, ix+batch_size)]
        pids = qry_slc['pid']
        names = qry_slc['name']
        tracks = qry_slc['tracks']

        # get initial prediction (for batch, 1k candidates)
        pids_ = [pid2ix[ix] for ix in pids.values]
        pred_raw_batch = np.argsort(U[pids_].dot(V.T), axis=1)[:, ::-1][:, :n_cand]

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
