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
    fire.Fire(main)
