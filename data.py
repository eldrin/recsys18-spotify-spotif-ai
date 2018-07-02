import glob
import os
from os.path import join, dirname, basename
from itertools import chain
from functools import partial
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

import torch

from util import read_data, read_hash

from tqdm import tqdm
import fire


FEATURE_COLS = [
    'acousticness', 'danceability', 'duration_ms', 'energy',
    'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
    'speechiness', 'tempo', 'time_signature', 'valence'
]


class DataPrepper:
    """"""
    def __init__(self, raw=True, subset=True, fullset=True):
        """ Preprocess given data to playlist-track-interaction lists

        Args:
            raw (bool): flag that indicates if raw data is processed
            subset (bool): flag that indicates if subset is genearted
            fullset (bool): flag that indicates if fullset is processed

        """
        self._scope = []
        self._is_raw = raw

        if subset:
            self._scope.append('subset')

        if fullset:
            self._scope.append('full')

    def process(self, data_root, out_root, challengeset_fn,
                n_train=50000, n_test=500):
        """ Prepare learning data for the project

        Args:
            data_root (str): top-level root dir for the raw data (mpd)
            out_root (str): top-level root dir for the processed data
            challengeset_fn (str): challengeset (json) file location
            n_train (int): number of training playlists
            n_test (int): number of testing playlists
        """
        # load setup
        processed_dir = data_root
        # render sub-paths
        subdirs = {}
        for k in self._scope:
            subdirs[k] = join(out_root, k)
            if not os.path.exists(subdirs[k]):
                os.makedirs(subdirs[k])

        if self._is_raw:
            # firing off the data prep process
            print('>>> Processing Raw Data!...')
            self._process_raw_data(data_root, out_root)

            print('>>> Processing Challenge Set!...')
            self._process_challenge_data(
                out_root, challengeset_fn,
                join(out_root, 'uniq_tracks.csv'))

        if 'full' in self._scope:
            print('>>> Preparing Fullset!...')
            self._get_full(
                join(out_root, 'playlist_tracks.csv'),
                join(out_root, 'uniq_playlists.csv'),
                join(out_root, 'uniq_tracks.csv'),
                join(out_root, 'uniq_artists.csv'),
                join(out_root, dirname(challengeset_fn),
                     basename(challengeset_fn).split('.json')[0] + '.csv'),
                subdirs['full'],
            )

        if 'subset' in self._scope:
            print('>>> Preparing Subset!...')
            self._get_subset(
                join(out_root, 'playlist_tracks.csv'),
                join(out_root, 'uniq_playlists.csv'),
                join(out_root, 'uniq_tracks.csv'),
                join(out_root, 'uniq_artists.csv'),
                subdirs['subset'], n_train, n_test
            )

    def _process_raw_data(self, data_root, out_path=None,
                          out_fn='playlist_tracks.csv',
                          track_hash_fn='uniq_tracks.csv',
                          artist_hash_fn='uniq_artists.csv',
                          playlist_hash_fn='uniq_playlists.csv'):
        """ Process raw data (MPD)  json files into csv files

        Args:
            data_root (str): path for the directory where the main data ('data/*.json') files staged
            out_fn (str): filename for the main assignment info
            track_hash_fn (str): track hash path
            artist_hash_fn (str): artist hash path
            playlist_hash_fn (str): playlist hash path
        """
        if out_path is not None and os.path.exists(out_path):
            out_fn = join(out_path, out_fn)
            track_hash_fn = join(out_path, track_hash_fn)
            artist_hash_fn = join(out_path, artist_hash_fn)
            playlist_hash_fn = join(out_path, playlist_hash_fn)

        _prepare_data(
            data_root, out_fn=out_fn, track_hash_fn=track_hash_fn,
            artist_hash_fn=artist_hash_fn, playlist_hash_fn=playlist_hash_fn
        )

    def _process_challenge_data(self, out_path, challengeset_fn, uniq_tracks_fn):
        """ Process raw challenge set into handy csv form
        Args:
            out_path (str): path to save output
            challengeset_fn (str): path for the raw file
            uniq_tracks_fn (str): track info dumped from self.procesS_raw_data

        TODO: need to be implemented
        """
        # preprocess fn
        fn = basename(challengeset_fn).split('.json')[0] + '.csv'
        fn = join(dirname(challengeset_fn), fn)

        # load relevant data
        data = json.load(open(challengeset_fn))
        track_hash = read_hash(uniq_tracks_fn)
        uri2ix = {k:int(v) for k, v in track_hash[[0, 2]].values}
        res = []  # (pid, ptitle, tid, ttitle)

        for playlist in data['playlists']:
            pid = playlist['pid']
            pttl = playlist['name'] if 'name' in playlist else ''
            if len(playlist['tracks']) == 0:
                res.append((pid, pttl, 0, 0))
            else:
                for track in playlist['tracks']:
                    tid = uri2ix[track['track_uri']]
                    tttl = track['track_name']

                    # add to list
                    res.append((pid, pttl, tid, tttl))

        # dump to file
        pd.DataFrame(res).to_csv(fn, index=None, sep='\t', encoding='utf-8')

    def _get_subset(self, playlist_track_fn, uniq_playlist_fn,
                   track_hash_fn, artist_hash_fn,
                   out_path, n_train=50000, n_test=500):
        """ get simulated subset out of the mpd training set """
        _subsample_dataset(
            r_fn=playlist_track_fn,
            uniq_playlist_fn=uniq_playlist_fn,
            track_hash_fn=track_hash_fn,
            artist_hash_fn=artist_hash_fn,
            out_path=out_path,
            b_fn=None, f_fn=None, p_fn=None,
            n_train=n_train, n_test=n_test)

    def _get_full(self, playlist_track_fn, uniq_playlist_fn,
                  track_hash_fn, artist_hash_fn, challengeset_fn,
                  out_path):
        """ get full dataset including challenge seeds as training set """
        _prepare_full_data(
            out_path=out_path,
            r_fn=playlist_track_fn,
            t_fn=challengeset_fn,
            uniq_playlist_fn=uniq_playlist_fn,
            track_hash_fn=track_hash_fn,
            artist_hash_fn=artist_hash_fn,
            b_fn=None, f_fn=None, p_fn=None)


def get_uniq_tracks_artists(fns):
    """"""
    all_trks = set()
    all_arts = set()
    trk_uri_nm = {}
    art_uri_nm = {}
    for fn in tqdm(fns, ncols=80):
        d = json.load(open(fn))
        trks = [
            (t['track_uri'], t['track_name'], t['artist_uri'], t['artist_name'])
            for pl in d['playlists']
            for t in pl['tracks']
        ]
        for t in trks:
            if t[0] not in trk_uri_nm:
                trk_uri_nm[t[0]] = t[1]
            if t[2] not in art_uri_nm:
                art_uri_nm[t[2]] = t[3]

        all_trks.update(map(lambda x: x[0], trks))
        all_arts.update(map(lambda x: x[2], trks))

    # get hash
    trk_hash = {v: k for k, v in enumerate(all_trks)}
    art_hash = {v: k for k, v in enumerate(all_arts)}

    return all_trks, trk_hash, trk_uri_nm, all_arts, art_hash, art_uri_nm


def _prepare_data(data_root, out_fn='playlist_tracks.csv',
                  track_hash_fn='uniq_tracks.csv',
                  artist_hash_fn='uniq_artists.csv',
                  playlist_hash_fn='uniq_playlists.csv'):
    """"""
    fns = glob.glob(join(data_root, 'data/*.json'))

    # get uniq tracks' hash
    (all_trks, trk_hash, trk_uri_nm,
     all_arts, art_hash, art_uri_nm) = get_uniq_tracks_artists(fns)
    print('Number of unique tracks: {:d}'.format(len(all_trks)))
    print('Number of unique tracks (in hash): {:d}'.format(len(trk_hash)))
    print('Number of unique artists: {:d}'.format(len(all_arts)))
    print('Number of unique artists (in hash): {:d}'.format(len(art_hash)))

    # write playlist - track tuple data to text file
    pid_nm = {}
    with open(join(data_root, out_fn), 'w') as f:
        for fn in tqdm(fns, ncols=80):
            for pl in json.load(open(fn))['playlists']:
                # save pid name
                if pl['pid'] not in pl:
                    pid_nm[pl['pid']] = pl['name']

                for t in pl['tracks']:
                    f.write('{:d},{:d},{:d}\n'.format(
                        int(pl['pid']),
                        int(trk_hash[t['track_uri']]),
                        int(art_hash[t['artist_uri']])))

    # write hash to file {URI (or PID) : assigned_ix}
    with open(join(data_root, track_hash_fn), 'wb') as f:
        for k, v in trk_hash.iteritems():
            f.write(
                u'{}\t{}\t{:d}\n'.format(k, trk_uri_nm[k], int(v))
                .encode('utf-8')
            )

    with open(join(data_root, artist_hash_fn), 'wb') as f:
        for k, v in art_hash.iteritems():
            f.write(
                u"{}\t{}\t{:d}\n".format(k, art_uri_nm[k], int(v))
                .encode('utf-8')
            )

    with open(join(data_root, playlist_hash_fn), 'wb') as f:
        for k, v in pid_nm.iteritems():
            f.write(
                u"{:d}\t{}\n".format(k, v).encode('utf-8')
            )


def fetch_k_tracks(R, k, playlist_ids, order=True):
    """"""
    r = []
    rt = []
    for j in playlist_ids:
        r_ = R[R[0]==j]  # fetch j_th playlist

        if order:  # first k tracks
            r_trn = r_.iloc[:k]  # first k tracks of playlist
            r_tst = r_.iloc[k:]  # rest tracks for testing
        else:  # random k tracks
            r__ = r_.sample(frac=1)
            r_trn = r__.iloc[:k]
            r_tst = r__.iloc[k:]

        # check up the shape & fix it
        if r_trn.ndim == 1:
            r_trn = r_trn.to_frame().T
        if r_tst.ndim == 1:
            r_tst = r_tst.to_frame().T

        # add in container
        r.append(r_trn)
        rt.append(r_tst)

    # aggregate
    r = pd.concat(r)
    rt = pd.concat(rt)

    return r, rt


def fetch_playlists_with_m_tracks(R, candidates_ids, l, m, tol=10):
    """
    R: interaction triplets (dataframe)
    candidates_ids: set of candidates playlist ids
    l: desired number of playlist fetched
    m: desired number of tracks included in the playlists (in average)
    tol: tolerence for the difference of # of tracks to the `m`
    """
    r = R[R[0].isin(set(candidates_ids))]  # candidate triplets
    n_tracks = r.groupby(0).count()[1]
    candidates = n_tracks[(n_tracks > m-tol) & (n_tracks <= m+tol)].index
    res = np.random.choice(candidates, l, replace=False)
    new_candidates = set(candidates_ids) - set(res)
    return res, new_candidates


def process_simulated_testset(R, A, B, n_train, n_test, tol=5):
    """"""
    # add interaction value (now all 1)
    R.loc[:, 2] = 1

    # get subset of playlists
    # pick up training playlist
    sampled_pl_tr = np.random.choice(R[0].unique(), n_train + n_test, replace=False)

    # split train / test
    # Rtr = R[R[0].isin(set(sampled_pl_tr))]

    # for split testset, do the post process for challenge set simulation
    Rts_tr = []  # the info going back to the training set (like challenge set seeds)
    Rts_ts = []  # info actually holding out for test
    n_chunk = int(n_test / 10)

    # pick up test play list
    # - now, need to pick ones that have at least more than `m` times more tracks
    #   than the seeds per each cases (i.e. more than `m` tracks if seed is 1)
    # - we will do the selection process iteratively, to make sure not picking
    #   already selected playlist picked again
    # - more specifically, the ratio between the seed:holdout in challenge set is:
    #   - seeds:  [ 0.0,  5.0,   5.0,  10.0,  10.0,   25.0,   25.0, 100.0, 100.0,   1.0]
    #   - E[h.o]: [28.6, 53.43, 58.07, 53.64, 53.68, 125.29, 126.77, 89.3,  87.57, 22.83]
    # ts_candidates = set(R[0].unique()) - set(sampled_pl_tr)
    ts_candidates = sampled_pl_tr

    # 1) only playlist titles
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=30, tol=tol)
    # r0 = Rts[Rts[0].isin(set(cands))]
    r0 = R[R[0].isin(set(cands))]
    Rts_ts.append(r0)
    r0t = r0.copy()
    r0t.loc[:, 2] = 0  # indication of no interaction observed
    Rts_tr.append(r0t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        1, len(cands), r0t[0].nunique()))

    # 2) the first track given
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=23, tol=tol)
    r1, r1t = fetch_k_tracks(R, 1, cands)
    Rts_tr.append(r1)
    Rts_ts.append(r1t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        2, len(cands), r1t[0].nunique()))

    # 3) first 5 tracks given
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=60, tol=tol)
    r2, r2t = fetch_k_tracks(R, 5, cands)
    Rts_tr.append(r2)
    Rts_ts.append(r2t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        3, len(cands), r2t[0].nunique()))

    # 4) first 5 tracks given | NO PLAYLIST TITLE
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=60, tol=tol)
    r3, r3t = fetch_k_tracks(R, 5, cands)
    Rts_tr.append(r3)
    Rts_ts.append(r3t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        4, len(cands), r3t[0].nunique()))

    # 5) first 10 tracks given
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=65, tol=tol)
    r4, r4t = fetch_k_tracks(R, 10, cands)
    Rts_tr.append(r4)
    Rts_ts.append(r4t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        5, len(cands), r4t[0].nunique()))

    # 6) first 10 tracks given | NO PLAYLIST TITLE
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=65, tol=tol)
    r5, r5t = fetch_k_tracks(R, 10, cands)
    Rts_tr.append(r5)
    Rts_ts.append(r5t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        6, len(cands), r5t[0].nunique()))

    # 7) first 25 tracks given
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=150, tol=tol)
    r6, r6t = fetch_k_tracks(R, 25, cands)
    Rts_tr.append(r6)
    Rts_ts.append(r6t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        7, len(cands), r6t[0].nunique()))

    # 8) random 25 tracks given
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=150, tol=tol)
    r7, r7t = fetch_k_tracks(R, 25, cands, order=False)
    Rts_tr.append(r7)
    Rts_ts.append(r7t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        8, len(cands), r7t[0].nunique()))

    # 9) first 100 tracks given
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=200, tol=tol)
    r8, r8t = fetch_k_tracks(R, 100, cands)
    Rts_tr.append(r8)
    Rts_ts.append(r8t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        9, len(cands), r8t[0].nunique()))

    # 10) random 100 tracks given
    cands, ts_candidates = fetch_playlists_with_m_tracks(
        R, ts_candidates, l=n_chunk, m=200, tol=tol)
    r9, r9t = fetch_k_tracks(R, 100, cands, order=False)
    Rts_tr.append(r9)
    Rts_ts.append(r9t)
    print('{:d}: cand({:d}) - fetched({:d})'.format(
        10, len(cands), r9t[0].nunique()))

    # aggregate test seeds into training set
    # Rtr = pd.concat([Rtr, pd.concat(Rts_tr, axis=0)], axis=0).drop_duplicates()
    Rtr = pd.concat([R[R[0].isin(set(ts_candidates))],
                     pd.concat(Rts_tr, axis=0)], axis=0).drop_duplicates()
    Rts = pd.concat(Rts_ts, axis=0)
    # Rts = Rts[Rts[1].isin(set(Rtr[1].unique()))]

    # find playlist contain the difference between train - test tracks
    uniq_tracks_test = set(Rts[1].unique())
    uniq_tracks_train = set(Rtr[1].unique())
    diff = uniq_tracks_test - uniq_tracks_train
    print('Before processing difference goes...')
    print(': {:d}'.format(len(diff)))

    # cut out tracks not introduced in train
    Rts = Rts[~Rts[1].isin(diff)]

    uniq_tracks_test = set(Rts[1].unique())
    uniq_tracks_train = set(Rtr[1].unique())
    diff = uniq_tracks_test - uniq_tracks_train
    print('After processing difference goes...')
    print(': {:d}'.format(len(diff)))


    # legacies.. (just for reference)
    print('Sampling artist...')
    uniq_trk = set(Rtr[1].unique())
    print('uniq_tracks:{:d}'.format(len(uniq_trk)))
    print(Rtr.shape)
    print(A.shape)
    print('getting indices')
    idx = A[1].isin(uniq_trk)
    print('filtering')
    a = A[idx].drop_duplicates()

    uniq_art = set(a[0].unique())
    if B is not None:
        b = B[B[0].isin(uniq_art)]
        b = b[b[1].isin(uniq_art)].drop_duplicates()
    else:
        b = B

    return Rtr, Rts, a, b


def _subsample_dataset(r_fn, uniq_playlist_fn, track_hash_fn, artist_hash_fn, out_path,
                       b_fn=None, f_fn=None, p_fn=None, n_train=10000, n_test=100):
    """"""
    print('Loading data!...')
    # 1. load main assignment file
    # (playlist, track, artist)
    D = pd.read_csv(r_fn, header=None, index_col=None)
    R = D[[0, 1]]  # playlist-track
    A = D[[2, 1]].drop_duplicates()  # artist-track
    A.columns = [0, 1]

    # load playlist titles
    playlists = pd.read_csv(uniq_playlist_fn, sep='\t',
                            index_col=None, header=None)
    playlist_id2ttl = dict(playlists.values)

    # load hashes to process
    with open(track_hash_fn) as f:
        track_hash = pd.DataFrame(
            map(lambda x: (x[0], x[1], int(x[2])),
                [l.replace('\n', '').split('\t') for l in f.readlines()])
        )
    track_uri2id = {k: v for k, v in track_hash[[0, 2]].values}
    track_id2uri = {v: k for k, v in track_uri2id.iteritems()}
    track_id2ttl = {k: t for k, t in track_hash[[2, 1]].values}

    with open(artist_hash_fn) as f:
        artist_hash = pd.DataFrame(
            map(lambda x: (x[0], x[1], int(x[2])),
                [l.replace('\n', '').split('\t') for l in f.readlines()])
        )
    artist_uri2id = {k: v for k, v in artist_hash[[0, 2]].values}
    artist_id2uri = {v: k for k, v in artist_uri2id.iteritems()}
    artist_id2ttl = {k: t for k, t in artist_hash[[2, 1]].values}

    if b_fn is not None and os.path.exists(b_fn):
        # (n_attr, n_attr)
        B = pd.read_csv(b_fn, header=None, index_col=None)  # attribute relationship
    else:
        B = None

    if f_fn is not None and os.path.exists(f_fn):
        # 2. load feature data
        F = pd.read_csv(f_fn, index_col='uri')  # audio feature [spotify]

        # (filling missing track and the NaN is manually done)
        # (code is just being here for the future report purpose)
        F = F.reindex(track_hash[0])  # filling missing tracks
        median = F.median()
        F = F.fillna(median)[FEATURE_COLS]

        # get hash
        feature_hash = {v: k for k, v in enumerate(F.index)}

        if p_fn is not None and os.path.exists(p_fn):

            # 3. load playlist / track meta features (i.e. popularity)
            # (currently only use the track metadata)
            P = pd.read_csv(p_fn, index_col='track_uri')
            P = P.reindex(F.index)
            median = P.median()
            P = P.fillna(median)
            F.loc[:, 'popularity'] = P['popularity']

    else:
        F = None

    print('Sampling & processing dataset!...')
    # sample by playlist
    data = process_simulated_testset(R, A, B, n_train, n_test)

    # (training_set, test_set, test_id, track_artist, artist_artist)
    rtr, rts, a, b = data
    r = pd.concat([rtr, rts], axis=0)  # for getting entire track list

    print('Re-indexing!...')
    # update hashing {old_ix: new_ix}
    new_pl_hash = {v: k for k, v in enumerate(r[0].unique())}
    new_tr_hash = {v: k for k, v in enumerate(r[1].unique())}
    new_ar_hash = {v: k for k, v in enumerate(a[0].unique())}

    # replace old index to new index
    rtr.loc[:, 0] = rtr[0].map(new_pl_hash)
    rtr.loc[:, 1] = rtr[1].map(new_tr_hash)
    rts.loc[:, 0] = rts[0].map(new_pl_hash)
    rts.loc[:, 1] = rts[1].map(new_tr_hash)

    a.loc[:, 0] = a[0].map(new_ar_hash)
    a.loc[:, 1] = a[1].map(new_tr_hash)

    if b is not None:
        b.loc[:, 0] = b[0].map(new_ar_hash)
        b.loc[:, 1] = b[1].map(new_ar_hash)

    new_pl_dict = {k: (playlist_id2ttl[k], k, v) for k, v in new_pl_hash.items()}
    new_tr_dict = {k: (track_id2ttl[k], track_id2uri[k], v) for k, v in new_tr_hash.items()}
    new_ar_dict = {k: (artist_id2ttl[k], artist_id2uri[k], v) for k, v in new_ar_hash.items()}

    # finally, make a playlist-artist matrix using playlist-track / artist-track mat
    track2artist = dict(a[[1, 0]].values)
    ctr = rtr.copy()
    cts = rts.copy()
    ctr.loc[:, 1] = ctr[1].map(track2artist)
    cts.loc[:, 1] = cts[1].map(track2artist)
    ctr = ctr.groupby([0, 1]).sum().reset_index()
    cts = cts.groupby([0, 1]).sum().reset_index()

    # print spec
    print('Rtr: ({:d}, {:d}) / sz:{:d}'.format(rtr[0].nunique(), rtr[1].nunique(), rtr.shape[0]))
    print('Rts: ({:d}, {:d}) / sz:{:d}'.format(rts[0].nunique(), rts[1].nunique(), rts.shape[0]))
    print('Ctr: ({:d}, {:d}) / sz:{:d}'.format(ctr[0].nunique(), ctr[1].nunique(), ctr.shape[0]))
    print('Cts: ({:d}, {:d}) / sz:{:d}'.format(cts[0].nunique(), cts[1].nunique(), cts.shape[0]))
    print('A: ({:d}, {:d}) / sz:{:d}'.format(a[0].nunique(), a[1].nunique(), a.shape[0]))
    if b is not None:
        print('B: ({:d}, {:d}) / sz:{:d}'.format(b[0].nunique(), b[1].nunique(), b.shape[0]))

    print('Saving!...')
    # save results
    rtr.to_csv(join(out_path, 'playlist_track_ss_train.csv'), header=None, index=None)
    rts.to_csv(join(out_path, 'playlist_track_ss_test.csv'), header=None, index=None)
    ctr.to_csv(join(out_path, 'playlist_artist_ss_train.csv'), header=None, index=None)
    cts.to_csv(join(out_path, 'playlist_artist_ss_test.csv'), header=None, index=None)
    a.to_csv(join(out_path, 'artist_track_ss.csv'), header=None, index=None)
    if b is not None:
        b.to_csv(join(out_path, 'artist_artist_ss.csv'), header=None, index=None)

    if F is not None:
        # normalize feature and save
        sclr = QuantileTransformer(1000, 'normal')

        ix2uri = {ix: uri for _, (_, uri, ix) in new_tr_dict.iteritems()}
        f = F.reindex([ix2uri[i] for i in range(len(ix2uri))])
        median = f.median()
        f = f.fillna(median)
        f = pd.get_dummies(f, columns=['key', 'mode', 'time_signature'])
        f.to_csv(join(out_path, 'track_audio_feature_ss.csv'), index=None)

        f = sclr.fit_transform(f)
        np.save(join(out_path, 'track_audio_feature_ss.npy'), f)

    # save hashes
    for name, dic in zip(['playlist', 'track', 'artist'], [new_pl_dict, new_tr_dict, new_ar_dict]):
        with open(join(out_path, '{}_hash_ss.csv'.format(name)), 'w') as f:
            for k, v in dic.iteritems():
                f.write("{:d}\t{}\t{}\t{}\n".format(int(k), v[0], v[1], v[2]))


def _prepare_full_data(out_path, r_fn, t_fn, uniq_playlist_fn,
                       track_hash_fn, artist_hash_fn,
                       b_fn=None, f_fn=None, p_fn=None):
    """
    Args:
        r_fn (str): training triplet (playlist, track, artist)
        t_fn (str): challenge set
        b_fn (str): artist-artist similarity triplet
        f_fn (str): spotify audio feature path
        p_fn (str): spotify track popularity path
    """
    print('Loading data!...')
    # 1. load main assignment file
    # (playlist, track, artist)
    D = pd.read_csv(r_fn, header=None, index_col=None)
    R = D[[0, 1]]  # playlist-track
    A = D[[2, 1]].drop_duplicates()  # artist-track
    A.columns = [0, 1]
    C_ = pd.read_csv(t_fn, index_col=None, sep='\t')  # challenge set
    C = C_[['0', '2']].copy()
    C.columns = [0, 1]
    C[2] = 1
    C.iloc[:1000, 2] = 0

    R[2] = 1
    R = pd.concat([R, C], axis=0)

    # load playlist titles
    playlists = pd.read_csv(uniq_playlist_fn, sep='\t',
                            index_col=None, header=None)
    playlist_id2ttl = dict(
        np.concatenate([playlists.values, C_[['0', '1']].values], axis=0))

    # load hashes to process
    # track_hash = pd.read_csv(track_hash_fn, header=None, index_col=None, sep='\t')
    with open(track_hash_fn) as f:
        track_hash = pd.DataFrame(
            map(lambda x: (x[0], x[1], int(x[2])),
                [l.replace('\n', '').split('\t') for l in f.readlines()])
        )
    track_uri2id = {k: v for k, v in track_hash[[0, 2]].values}
    track_id2uri = {v: k for k, v in track_uri2id.iteritems()}
    track_id2ttl = {k: t for k, t in track_hash[[2, 1]].values}

    # artist_hash = pd.read_csv(artist_hash_fn, header=None, index_col=None, sep='\t')
    with open(artist_hash_fn) as f:
        artist_hash = pd.DataFrame(
            map(lambda x: (x[0], x[1], int(x[2])),
                [l.replace('\n', '').split('\t') for l in f.readlines()])
        )
    artist_uri2id = {k: v for k, v in artist_hash[[0, 2]].values}
    artist_id2uri = {v: k for k, v in artist_uri2id.iteritems()}
    artist_id2ttl = {k: t for k, t in artist_hash[[2, 1]].values}

    # (n_attr, n_attr)
    if b_fn is not None and os.path.exists(b_fn):
        B = pd.read_csv(b_fn, header=None, index_col=None)  # attribute relationship
    else:
        B = None

    if f_fn is not None and os.path.exists(f_fn):
        # 2. load feature data
        F = pd.read_csv(f_fn, index_col='uri')  # audio feature [spotify]

        # (filling missing track and the NaN is manually done)
        # (code is just being here for the future report purpose)
        F = F.reindex(track_hash[0])  # filling missing tracks
        median = F.median()
        F = F.fillna(median)[FEATURE_COLS]

        # get hash
        feature_hash = {v: k for k, v in enumerate(F.index)}

        if p_fn is not None and os.path.exists(p_fn):

            # 3. load playlist / track meta features (i.e. popularity)
            # (currently only use the track metadata)
            P = pd.read_csv(p_fn, index_col='track_uri')
            P = P.reindex(F.index)
            median = P.median()
            P = P.fillna(median)
            F.loc[:, 'popularity'] = P['popularity']

    else:
        F = None

    print('Processing dataset!...')
    # (training_set, test_set, test_id, track_artist, artist_artist)
    rtr, a, b = R, A, B

    print('Re-indexing!...')
    # update hashing {old_ix: new_ix}
    new_pl_hash = {v: k for k, v in enumerate(rtr[0].unique())}
    new_tr_hash = {v: k for k, v in enumerate(rtr[1].unique())}
    new_ar_hash = {v: k for k, v in enumerate(a[0].unique())}

    # replace old index to new index
    rtr.loc[:, 0] = rtr[0].map(new_pl_hash)
    rtr.loc[:, 1] = rtr[1].map(new_tr_hash)

    a.loc[:, 0] = a[0].map(new_ar_hash)
    a.loc[:, 1] = a[1].map(new_tr_hash)
    if b is not None:
        b.loc[:, 0] = b[0].map(new_ar_hash)
        b.loc[:, 1] = b[1].map(new_ar_hash)

    # {old_ix: (title, uri, new_id?)}
    new_pl_dict = {k: (playlist_id2ttl[k], k, v) for k, v in new_pl_hash.items()}
    new_tr_dict = {k: (track_id2ttl[k], track_id2uri[k], v) for k, v in new_tr_hash.items()}
    new_ar_dict = {k: (artist_id2ttl[k], artist_id2uri[k], v) for k, v in new_ar_hash.items()}

    # finally, make a playlist-artist matrix using playlist-track / artist-track mat
    track2artist = dict(a[[1, 0]].values)
    ctr = rtr.copy()
    ctr.loc[:, 1] = ctr[1].map(track2artist)
    ctr = ctr.groupby([0, 1]).sum().reset_index()

    # print spec
    print('Rtr: ({:d}, {:d}) / sz:{:d}'.format(rtr[0].nunique(), rtr[1].nunique(), rtr.shape[0]))
    print('Ctr: ({:d}, {:d}) / sz:{:d}'.format(ctr[0].nunique(), ctr[1].nunique(), ctr.shape[0]))
    print('A: ({:d}, {:d}) / sz:{:d}'.format(a[0].nunique(), a[1].nunique(), a.shape[0]))
    if b is not None:
        print('B: ({:d}, {:d}) / sz:{:d}'.format(b[0].nunique(), b[1].nunique(), b.shape[0]))

    print('Saving!...')
    # save results
    rtr.to_csv(join(out_path, 'playlist_track_train.csv'), header=None, index=None)
    ctr.to_csv(join(out_path, 'playlist_artist_train.csv'), header=None, index=None)
    a.to_csv(join(out_path, 'artist_track.csv'), header=None, index=None)
    if b is not None:
        b.to_csv(join(out_path, 'artist_artist.csv'), header=None, index=None)

    if F is not None:
        # normalize feature and save
        sclr = QuantileTransformer(1000, 'normal')

        ix2uri = {ix: uri for _, (_, uri, ix) in new_tr_dict.iteritems()}
        f = F.reindex([ix2uri[i] for i in range(len(ix2uri))])
        median = f.median()
        f = f.fillna(median)
        f = pd.get_dummies(f, columns=['key', 'mode', 'time_signature'])
        f.to_csv(join(out_path, 'track_audio_feature.csv'), index=None)

        f = sclr.fit_transform(f)
        np.save(join(out_path, 'track_audio_feature.npy'), f)

    # save hashes
    for name, dic in zip(['playlist', 'track', 'artist'], [new_pl_dict, new_tr_dict, new_ar_dict]):
        with open(join(out_path, '{}_hash.csv').format(name), 'w') as f:
            for k, v in dic.iteritems():
                f.write("{:d}\t{}\t{}\t{}\n".format(int(k), v[0], v[1], v[2]))


def _load_embeddings(config):
    """"""
    embs = {
        k: np.load(fn) for k, fn
        in config['path']['embeddings'].iteritems()
    }
    return embs


def _load_data(config):
    """ returns main tuples and track-artist map """
    dat = {
        k: pd.read_csv(fn, header=None) for k, fn
        in config['path']['data'].iteritems()
        if k not in {'playlists', 'tracks', 'artists'}
    }
    # dat['main'].columns = ['playlist', 'track']
    dat['train'].columns = ['playlist', 'track', 'value']  # set columns name
    # dat['train'] = dat['train'][dat['train']['value'] == 1]

    if 'test' in dat:
        dat['test'].columns = ['playlist', 'track', 'value']
        return (
            None, dat['train'], dat['test'],
            dict(dat['artist2track'][[1, 0]].values)
        )
    else:
        return (
            None, dat['train'], None,
            dict(dat['artist2track'][[1, 0]].values)
        )


class MPDSampler:
    """"""
    def __init__(self, config, verbose=False):
        """"""
        _, self.train, self.test, self.track2artist = _load_data(config)
        self.triplet = self.train

        self.n_playlists = self.triplet['playlist'].nunique()
        self.triplet = self.triplet[self.triplet['value'] == 1]
        if self.test is None:
            self.items = self.triplet['track'].unique()
        else:
            self.items = list(np.unique(
                np.concatenate([self.triplet['track'].unique(),
                                self.test['track'].unique()])))
        self.n_tracks = len(self.items)
        self.num_interactions = self.triplet.shape[0]
        self.batch_size = config['hyper_parameters']['batch_size']
        self.is_weight = config['hyper_parameters']['sample_weight']
        self.weight_pow = config['hyper_parameters']['sample_weight_power']
        self.threshold = config['hyper_parameters']['sample_threshold']
        self.with_context = config['hyper_parameters']['with_context']
        self.context_size = [1, 5, 25, 50, 100]
        self.context_shuffle = [False, True]
        self.w0 = 10  # positive weight
        self.c0 = 1  # negative initial weight

        # prepare positive sample pools
        self.pos_tracks = dict(self.triplet.groupby('playlist')['track'].apply(set))

        if self.test is not None:
            self.pos_tracks_t = dict(self.test.groupby('playlist')['track'].apply(set))

        if self.is_weight:
            # check word (track) frequency
            track_count = self.triplet.groupby('track').count()['playlist']
            f_w_dict = track_count.to_dict()
            f_w = [f_w_dict[t] if t in f_w_dict else 0 for t in self.items]

            # Sub-sampling
            f_w_a = np.array(f_w)**.5
            c = f_w_a / f_w_a.sum() * self.c0
            self.items_dict = dict(enumerate(self.items))
            self.weight = dict(zip(self.items, c))
            self.pos_weight = self.w0

        else:
            # self.items = self.triplet['track'].unique()
            self.items_dict = dict(enumerate(self.items))
            self.weight = dict(zip(self.items, [1] * len(self.items)))
            self.pos_weight = self.w0

        self.neg = config['hyper_parameters']['neg_sample']
        self.verbose = verbose

    # @background(max_prefetch=100)
    def generator(self):
        """"""
        if self.verbose:
            M = tqdm(self.triplet.sample(frac=1).values, ncols=80)
        else:
            M = self.triplet.sample(frac=1).values

        batch, neg, conf = [], [], []
        for u, i, v in M:

            if v == 0:
                continue

            # positive sample / yield
            pos_i = self.pos_tracks[u]
            conf.append(self.pos_weight)

            # add context
            if self.with_context:
                context = pos_i - set([i])
                if len(context) > 0:
                    N = filter(lambda x: x <= len(context), self.context_size)
                    context = np.random.choice(
                        list(context), np.random.choice(N), replace=False)
                else:  # randomly pick popular songs...
                    context = np.random.choice(100000, 5, False)
            else:
                context = 0

            # draw negative samples (for-loop)
            for k in xrange(self.neg):
                j_ = self.items_dict[np.random.choice(self.n_tracks)]
                while j_ in pos_i:
                    j_ = self.items_dict[np.random.choice(self.n_tracks)]
                # negtive sample has 0 interaction (conf==1)
                neg.append(j_)
                conf.append(self.weight[j_])

            # prepare batch containor
            batch.append((u, i, neg, context, conf))
            neg, context, conf = [], [], []

            # batch.append(batch_)
            if len(batch) >= self.batch_size:
                yield batch
                # reset containors
                batch = []


def get_ngram(word, n=3):
    """"""
    w = '#' + str(word) + '#'
    return [w[i:i+n] for i in range(len(w)-n+1)]


def get_unique_ngrams(words, n=3, stopper='#'):
    """"""
    ngrams = map(partial(get_ngram, n=n), words)
    uniq_ngrams = list(set(chain.from_iterable(ngrams)))  # flatten
    return uniq_ngrams


def transform_id2ngram_id(ids, title_dict, ngram_dict, n=3):
    """
    transform a index (either playlist or track)
    into sequence of ngram_id
    """
    out = []
    for i in ids:
        out.append(
            [ngram_dict[w] for w in get_ngram(title_dict[i], n=n)]
        )
    return out


class SeqTensor:
    def __init__(self, seq, title_dict=None, ngram_dict=None):
        """"""
        # process seq
        if title_dict is None and ngram_dict is None:
            seq_ngram = seq
        else:  # process on-the-go
            seq_ngram = transform_id2ngram_id(seq, title_dict, ngram_dict)
        lengths = map(len, seq_ngram)
        max_len = max(lengths)
        lengths = torch.cuda.LongTensor(lengths)
        X_pl = torch.cuda.LongTensor(
            [a + [0] * (max_len - len(a)) for a in seq_ngram])
        length_sorted, ind = lengths.sort(descending=True)
        _, self.unsort_ind = ind.sort()

        # assign properties
        self.seq = X_pl[ind]
        self.lengths = length_sorted
        self.ind = ind

    def unsort(self, h):
        """"""
        return h[self.unsort_ind]


if __name__ == "__main__":
    fire.Fire(DataPrepper)
