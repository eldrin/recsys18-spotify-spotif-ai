import json
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
import cPickle as pkl

from util import read_hash

from tqdm import tqdm
import fire


def simple_rank_score(rank):
    """"""
    return 1. / np.log(rank + 1.)


def get_score(distance, ranking_score_template):
    """"""
    return np.log(
        ranking_score_template[distance.argsort(1).argsort(1)]
    ).mean(0)


def from_mf(config_fn, alpha=5, cutoff=50000, dist_fnc='cosine'):
    """"""
    # load config
    config = json.load(open(config_fn))

    # parse
    out_fn = config["out_fn"]
    user_factor_fn = config["user_factor_fn"]
    item_factor_fn = config["item_factor_fn"]
    artist_factor_fn = config["artist_factor_fn"]
    challenge_set_fn = config["challenge_set_fn"]
    track_hash_fn = config["track_hash_fn"]
    playlist_hash_fn = config["playlist_hash_fn"]
    artist_track_fn = config["artist_track_fn"]

    # load track hash
    pl_hash = read_hash(playlist_hash_fn)
    pid2pl = {int(v):int(k) for k, v in pl_hash[[3, 2]].values}

    tr_hash = read_hash(track_hash_fn)
    uri2tr = {v:int(k) for k, v in tr_hash[[3, 2]].values}
    tr2uri = {v:k for k, v in uri2tr.iteritems()}

    # load artist hash
    art2tr = pd.read_csv(artist_track_fn, header=None)
    tr2art = dict(art2tr[[1, 0]].values)

    # load MF model
    P = np.load(user_factor_fn)
    Q = np.load(item_factor_fn)

    # reranking source
    W = np.load(artist_factor_fn)
    W = W[[tr2art[i] for i in xrange(Q.shape[0])]]

    # load seeds
    challenge_set= json.load(open(challenge_set_fn))
    seeds = dict(
        map(
            lambda x:
            (x['pid'], set(map(lambda y: y['track_uri'], x['tracks']))),
            challenge_set['playlists']
        )
    )

    # pre-calc rank score
    r_s = simple_rank_score(np.arange(1., Q.shape[0]+1))
    r_s /= r_s.sum()

    # running naive-reranking
    # re-write re-ranked result
    f = open(out_fn, 'w')
    f.write("team_info,creative,spotif.ai,J.H.Kim@tudelft.nl\n")
    new_rec = {}
    for pid, seed in tqdm(seeds.iteritems(), total=len(seeds), ncols=80):

        D = -P[pid2pl[pid]].dot(Q.T)[None]
        org_score = np.log(r_s[D.argsort(axis=1).argsort(axis=1)])

        if len(seed) > 0:
            r = org_score[0].argsort()[::-1][:cutoff]

            # process artist-base re-ranking
            wq = W[[tr2art[uri2tr[uri]] for uri in seed]]
            qq = Q[[uri2tr[uri] for uri in seed]]

            # get distance mat
            aux_scores = []
            aux_scores.append(get_score(dist.cdist(wq, W[r], metric=dist_fnc), r_s))
            aux_scores.append(get_score(dist.cdist(qq, Q[r], metric=dist_fnc), r_s))

            # get averaged re-rank score
            new_score = org_score[0, r] + alpha * np.mean(aux_scores, axis=0)
            new_rank = r[np.argsort(new_score)[::-1]]
        else:
            new_rank = org_score[0].argsort()[::-1][cutoff]

        # get new recommendation
        track_uris = filter(
            lambda rec: rec not in seed,
            [tr2uri[j] for j in new_rank[:600]]
        )[:500]

        # write down to file
        f.write("{:d},".format(pid) + ",".join(track_uris) + "\n")
    f.close()


def main(init_rank_path, challenge_query_pkl, artist_factor_fn,
         track_hash_fn, artist_track_fn):
    """"""

    # loading initial rank
    init_rank = pd.read_csv(init_rank_path, skiprows=1,
                            header=None, sep=',', index_col=0)
    # load track hash
    tr_hash = read_hash(track_hash_fn)
    uri2tr = {v:int(k) for k, v in tr_hash[[3, 2]].values}

    # load artist hash
    art2tr = pd.read_csv(artist_track_fn, header=None)
    tr2art = dict(art2tr[[1, 0]].values)

    # reranking source
    W = np.load(artist_factor_fn)

    # load seeds
    seeds = pkl.load(open(challenge_query_pkl))

    # pre-calc rank score
    r_s = simple_rank_score(np.arange(1., init_rank.shape[1] + 1))
    r_s /= r_s.sum()

    # running naive-reranking
    new_rec = {}
    for pid, recs in tqdm(init_rank.iterrows(), total=init_rank.shape[0], ncols=80):

        if len(seeds[pid]) > 0:
            # process artist-base re-ranking
            w = W[[tr2art[uri2tr[uri]] for uri in recs]]
            wq = W[[tr2art[uri2tr[uri]] for uri in seeds[pid]]]

            assert wq.shape[0] > 0

            # get distance mat
            D = dist.cdist(wq, w)

            # get averaged re-rank score
            new_score = np.log(r_s[D.argsort(axis=1).argsort(axis=1)] + 1e-10).sum(axis=0)
            new_rank = np.argsort(new_score)[::-1][:500] + 1

            new_rec_ = init_rank.loc[pid].reindex(new_rank)
            # print(new_rec_.isna().sum())
            new_rec[pid] = new_rec_.tolist()
        else:
            new_rec[pid] = init_rank.loc[pid].tolist()[:500]

    # re-write re-ranked result
    out_fn = init_rank_path.split('.csv')[0] + '_rerank.csv'
    with open(out_fn, 'w') as f:
        f.write("team_info,creative,spotif.ai,J.H.Kim@tudelft.nl\n")
        for pid, track_uris in tqdm(new_rec.iteritems(), ncols=80):
            # print(track_uris)
            f.write("{:d},".format(pid) + ",".join(track_uris) + "\n")


if __name__ == "__main__":
    # fire.Fire(main)
    fire.Fire(from_mf)
