import glob
import os
import json

import pandas as pd
import numpy as np

from util import read_data

from tqdm import tqdm
import fire


def get_uniq_tracks_artists(fns):
    """"""
    all_trks = set()
    all_arts = set()
    art_uri_nm = {}
    for fn in tqdm(fns, ncols=80):
        d = json.load(open(fn))
        trks = [
            (t['track_uri'], t['artist_uri'], t['artist_name'])
            for pl in d['playlists']
            for t in pl['tracks']
        ]
        for t in trks:
            if t[1] not in art_uri_nm:
                art_uri_nm[t[1]] = t[2]
        all_trks.update(map(lambda x: x[0], trks))
        all_arts.update(map(lambda x: x[1], trks))

    # get hash
    trk_hash = {v: k for k, v in enumerate(all_trks)}
    art_hash = {v: k for k, v in enumerate(all_arts)}

    return all_trks, trk_hash, all_arts, art_hash, art_uri_nm


def prepare_data(data_root, out_fn='playlist_tracks.csv',
                 track_hash_fn='uniq_tracks.csv', artist_hash_fn='uniq_artists.csv'):
    """"""
    fns = glob.glob(os.path.join(data_root, 'data/*.json'))

    # get uniq tracks' hash
    _, trk_hash, _, art_hash, art_uri_nm = get_uniq_tracks_artists(fns)

    # write playlist - track tuple data to text file
    with open(os.path.join(data_root, out_fn), 'w') as f:
        for fn in tqdm(fns, ncols=80):
            for pl in json.load(open(fn))['playlists']:
                for t in pl['tracks']:
                    f.write('{:d},{:d},{:d}\n'.format(
                        int(pl['pid']),
                        int(trk_hash[t['track_uri']]),
                        int(art_hash[t['artist_uri']])))

    # write hash to file
    with open(os.path.join(data_root, track_hash_fn), 'w') as f:
        for k, v in trk_hash.iteritems():
            f.write('{},{:d}\n'.format(k, int(v)))

    with open(os.path.join(data_root, artist_hash_fn), 'wb') as f:
        for k, v in art_hash.iteritems():
            f.write(
                u"{}\t{}\t{:d}\n".format(k, art_uri_nm[k], int(v))
                .encode('utf-8')
            )


def subsample_dataset(r_fn, b_fn, track_hash_fn, artist_hash_fn, n_pl=10000, ratio=1):
    """"""
    print('Loading data!...')
    # (n_user, n_item)
    D = pd.read_csv(r_fn, header=None, index_col=None)
    R = D[[0, 1]]

    # (n_attr, n_item)
    # A = pd.read_csv(a_fn, header=None, index=None, delimiter='\t')
    A = D[[2, 1]]
    A.columns = [0, 1]

    # (n_attr, n_attr)
    B = pd.read_csv(b_fn, header=None, index_col=None)

    # load hashes to process
    track_hash = pd.read_csv(track_hash_fn, header=None, index_col=None)
    track_hash = {v: k for k, v in track_hash.as_matrix()}
    artist_hash = pd.read_csv(artist_hash_fn, header=None, index_col=None, sep='\t')
    artist_hash = {v: k for k, v in artist_hash[[0, 2]].as_matrix()}

    print('Sample dataset!...')
    # sample interaction
    # sample by playlist
    sampled_pl = set(np.random.choice(R[0].unique(), n_pl, replace=False))
    R = R[R[0].isin(sampled_pl)]

    n = R.shape[0]
    bound = int(n * ratio)
    rnd_idx = np.random.choice(n, bound, replace=False)
    r = R.iloc[rnd_idx[:bound]]

    uniq_trk = set(r[1].unique())
    a = A[A[1].isin(uniq_trk)].drop_duplicates()

    uniq_art = set(a[0].unique())
    b = B[B[0].isin(uniq_art)]
    b = b[b[1].isin(uniq_art)].drop_duplicates()

    print('Re-indexing!...')
    # update hashing {old_ix: new_ix}
    new_pl_hash = {v: k for k, v in enumerate(r[0].unique())}
    new_tr_hash = {v: k for k, v in enumerate(r[1].unique())}
    new_ar_hash = {v: k for k, v in enumerate(a[0].unique())}

    # replace old index to new index
    # r.replace({0: new_pl_hash, 1: new_tr_hash})
    r.loc[:, 0] = r[0].map(new_pl_hash)
    r.loc[:, 1] = r[1].map(new_tr_hash)
    # a.replace({0: new_ar_hash, 1: new_tr_hash})
    a.loc[:, 0] = a[0].map(new_ar_hash)
    a.loc[:, 1] = a[1].map(new_tr_hash)
    # b.replace({0: new_ar_hash, 1: new_ar_hash})
    b.loc[:, 0] = b[0].map(new_ar_hash)
    b.loc[:, 1] = b[1].map(new_ar_hash)

    new_pl_hash_ = {k: v for k, v in enumerate(new_pl_hash)}
    new_tr_hash_ = {track_hash[k]: v for k, v in enumerate(new_tr_hash)}
    new_ar_hash_ = {artist_hash[k]: v for k, v in enumerate(new_ar_hash)}

    # print spec
    print('R: ({:d}, {:d}) / sz:{:d}'.format(r[0].nunique(), r[1].nunique(), r.shape[0]))
    print('A: ({:d}, {:d}) / sz:{:d}'.format(a[0].nunique(), a[1].nunique(), a.shape[0]))
    print('B: ({:d}, {:d}) / sz:{:d}'.format(b[0].nunique(), b[1].nunique(), b.shape[0]))

    print('Saving!...')
    # save results
    r.to_csv('/mnt/bulk/recsys18/playlist_track_ss.csv', header=None, index=None)
    a.to_csv('/mnt/bulk/recsys18/artist_track_ss.csv', header=None, index=None)
    b.to_csv('/mnt/bulk/recsys18/artist_artist_ss.csv', header=None, index=None)

    # save hashes
    for name, dic in zip(['playlist', 'track', 'artist'], [new_pl_hash, new_tr_hash, new_ar_hash]):
        with open('/mnt/bulk/recsys18/{}_hash_ss.csv'.format(name), 'w') as f:
            for k, v in dic.iteritems():
                f.write("{},{:d}\n".format(k, int(v)))


if __name__ == "__main__":
    # fire.Fire(prepare_data)
    fire.Fire(subsample_dataset)
