import os
from os.path import join
import sys
sys.path.append(join(os.getcwd(), 'RecsysChallengeTools'))
import subprocess

import json
import tempfile

from data import DataPrepper
from pretrain_cf import main as pretrain_cf
from train_rnn import main as train_rnn
from post_process import main as post_process
from prepare_submission import main as prepare_submission

from verify_submission import verify_submission

import fire

MF_MODEL_NAME = 'wrmf{:d}'
RNN_MODEL_NAME = 'titlernn{:d}'
MFRNN_PLAYLIST_MODEL_NAME = 'mf_rnn{:d}'


def main(data_root, challengeset_fn, out_root, n_factors=10, use_gpu=False):
    """"""
    # setup model names
    mf_name = MF_MODEL_NAME.format(n_factors)
    rnn_name = RNN_MODEL_NAME.format(n_factors)
    mfrnn_pl_name = MFRNN_PLAYLIST_MODEL_NAME.format(n_factors)

    print('>>> 1. Processing dataset!...')
    # only process fullset (we don't need subset now)
    data = (DataPrepper(subset=False)
            .process(data_root, out_root, challengeset_fn))

    print('>>> 2. Pre-training MF (WRMF) model!...')
    pretrain_cf(
        train_fn=join(out_root, "full", "playlist_track_train.csv"),
        r=n_factors, model_out_root=out_root,
        model_name=mf_name
    )

    # print('>>> 3. Training RNN (Char-Ngram-LSTM) model!...')
    # > build tmp config
    rnn_conf = {
        'path':{
            'embeddings':{
                'U': join(out_root, '{}_U.npy'.format(mf_name)),
                'V': join(out_root, '{}_V.npy'.format(mf_name))
            },
            'data':{
                'playlists': join(out_root, 'full', 'playlist_hash.csv'),
                'tracks': join(out_root, 'full', 'track_hash.csv'),
                'artists': join(out_root, 'full', 'artist_hash.csv'),
                'train': join(out_root, 'full', 'playlist_track_train.csv'),
                'artist2track': join(out_root, 'full', 'artist_track.csv')
            },
            'model_out':{
                'U': join(out_root, '{}_U.npy'.format(rnn_name)),
                'V': join(out_root, '{}_V.npy'.format(rnn_name)),
                "rnn": join(out_root, "rnn_checkpoint.pth.tar")
            }
        },
        'hyper_parameters':{
            "early_stop": True,
            "use_gpu": use_gpu,
            "eval_while_fit": True,
            "sample_weight": False,
            "sample_weight_power": 0.75,
            "sample_threshold": 1e-6,
            "with_context": False,
            "num_epochs": 100,
            "neg_sample": 4,
            "loss": "all",
            "learn_rate": 0.005,
            "batch_size": 1024,
            "n_embedding": 300,
            "n_out_embedding": n_factors,
            "n_hid": 1000,
            "n_layers": 1,
            "drop_out": 0,
            "l2": 1e-4,
            "alpha": 0,
            "ngram_n": 3
        },
    }
    with tempfile.NamedTemporaryFile(suffix='.json') as tmpf:
        json.dump(rnn_conf, open(tmpf.name, 'w'))
        train_rnn(tmpf.name)

    # 4. post process (merge two playlist factors (mf/rnn))
    print('>>> 4. Combining MF-RNN playlist factors!...')
    post_process(
        rnn_conf['path']['embeddings']['U'],
        rnn_conf['path']['model_out']['U'],
        join(out_root, '{}_U.npy'.format(mfrnn_pl_name)),
        rnn_conf['path']['data']['train']
    )

    # 5. prepare submission file
    print('>>> 5. Preparing submission file!!...')
    prep_conf = {
        'path':{
            'models':{
                'als_rnn':{
                    'name':'main',
                    'P': join(out_root, '{}_U.npy'.format(mfrnn_pl_name)),
                    'Q': join(out_root, rnn_conf['path']['embeddings']['V']),
                    'importance':1,
                    'logistic': False
                }
            },
            'data':{
                'playlists': join(out_root, "full", 'playlist_hash.csv'),
                'tracks': join(out_root, "full", 'track_hash.csv'),
                'train': join(out_root, "full", 'playlist_track_train.csv'),
                'challenge_set': challengeset_fn
            },
            'output': './data/wrmf_titlernn_spotif_ai.csv'
        }
    }
    with tempfile.NamedTemporaryFile(suffix='.json') as tmpf:
        json.dump(prep_conf, open(tmpf.name, 'w'))
        prepare_submission(tmpf.name)

    # verify if the submission file is valid
    print('>>>>>> verifying submission file...')
    errors = verify_submission(challengeset_fn, prep_conf['path']['output'])
    assert errors == 0
    print('>>>>>>>>> No errors found!...')

    print
    print('>>>>>> All process finished!!')
    print
    print


if __name__ == "__main__":
    fire.Fire(main)
