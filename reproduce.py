import os
import tempfile
from os.path import join
from data import DataPrepper
from pretrain_cf import main as pretrain_cf
from train_rnn import main as train_rnn
from post_process import main as post_process
from prepare_submission import main as prepare_submission

import fire


def main(data_root, challengeset_fn, out_root, n_factors=10):
    """"""
    print('>>> 1. Processing dataset!...')
    data = DataPrepper().process(data_root, out_root, challengeset_fn)

    print('>>> 2. Pre-training MF (WRMF) model!...')
    pretrain_cf(
        train_fn=join(out_root, "full", "playlist_track_train.csv"),
        r=n_factors, model_out_root=out_root, model_name='wrmf'
    )

    print('>>> 3. Training RNN (Char-Ngram-LSTM) model!...')
    # > build tmp config
    rnn_conf = {
        'path':{
            'embeddings':{
                'U': join(out_root, 'wrmf_U.npy'),
                'V': join(out_root, 'wrmf_V.npy')
            },
            'data':{
                'playlists': join(out_root, 'playlist_hash.csv'),
                'tracks': join(out_root, 'track_hash.csv'),
                'artists': join(out_root, 'artist_hash.csv'),
                'train': join(out_root, 'playlist_track_train.csv'),
                'artist2track': join(out_root, 'artist_track.csv')
            },
            'model_out':{
                'U': join(out_root, 'titlernn{:d}_U.npy'.format(n_factors)),
                'V': join(out_root, 'titlernn{:d}_V.npy'.format(n_factors)),
                "rnn": join(out_root, "rnn_checkpoint.pth.tar")
            }
        },
        'hyper_parameters':{
            "early_stop": true,
            "use_gpu": false,
            "eval_while_fit": true,
            "sample_weight": false,
            "sample_weight_power": 0.75,
            "sample_threshold": 1e-6,
            "with_context": false,
            "num_epochs": 100,
            "neg_sample": 4,
            "track_emb": "artist_ngram",
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
        json.dump(open(tmpf.name, 'w'))
        train_rnn(tmpf.name)

    # 4. post process (merge two playlist factors (mf/rnn))
    print('>>> 4. Combining MF-RNN playlist factors!...')
    post_process(
        rnn_conf['path']['embeddings']['U'],
        rnn_conf['model_out']['U'],
        join(out_root, 'mf_rnn_U.npy'),
        rnn_conf['path']['data']['train']
    )

    # 5. prepare submission file
    print('>>> 5. Preparing submission file!!...')
    prep_conf = {
        'path':{
            'models':{
                'als_rnn':{
                    'name':'main',
                    'P': join(out_root, 'mf_rnn_U.npy'),
                    'Q': join(out_root, rnn_conf['path']['embeddings']['V']),
                    'importance':1,
                    'logistic': False
                }
            },
            'data':{
                'playlists': join(out_root,'playlist_hash.csv'),
                'tracks': join(out_root, 'track_hash.csv'),
                'train': join(out_root, 'playlist_track_train.csv'),
                'challenge_set': challengeset_fn
            },
            'output': './data/wrmf_titlernn_spotif_ai.csv'
        }
    }
    with tempfile.NamedTemporaryFile(suffix='.json') as tmpf:
        json.dump(open(tmpf.name, 'w'))
        prepare_submission(tmpf.name)


if __name__ == "__main__":
    fire.Fire(main)
