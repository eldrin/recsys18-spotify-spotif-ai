import glob
import os
import json

import pandas
import numpy as np

from tqdm import tqdm
import fire


def get_uniq_tracks(fns):
    """"""
    all_trks = set()
    for fn in tqdm(fns, ncols=80):
        d = json.load(open(fn))
        trks = [
            t['track_uri']
            for pl in d['playlists']
            for t in pl['tracks']
        ]
        all_trks.update(trks)

    # get hash
    trk_hash = {v: k for k, v in enumerate(all_trks)}

    return all_trks, trk_hash


def prepare_data(data_root, out_fn='playlist_tracks.csv', hash_fn='uniq_tracks.csv'):
    """"""
    fns = glob.glob(os.path.join(data_root, 'data/*.json'))

    # get uniq tracks' hash
    _, trk_hash = get_uniq_tracks(fns)

    # write playlist - track tuple data to text file
    with open(os.path.join(data_root, out_fn), 'w') as f:
        for fn in tqdm(fns, ncols=80):
            for pl in json.load(open(fn))['playlists']:
                for t in pl['tracks']:
                    f.write('{:d},{:d}\n'.format(
                        int(pl['pid']), int(trk_hash[t['track_uri']])))

    # write hash to file
    with open(os.path.join(data_root, hash_fn), 'w') as f:
        for k, v in trk_hash.iteritems():
            f.write('{},{:d}\n'.format(k, int(v)))


if __name__ == "__main__":
    fire.Fire(prepare_data)


