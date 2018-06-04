from functools import partial
from torch import nn
from torch import optim

CONFIG = {
    'path':{
        'embeddings':{
            'U': './data/wrmf_U.npy',
            'V': './data/wrmf_V.npy',
            'W': './data/wrmf_W.npy',
            'X': '/mnt/bulk/recsys18/track_audio_feature_ss.npy',
            'track_word': './data/w_emb_skipgram_track_ttl_gensim.npy',
            'track_id2word': './data/track_id2word.pkl'
        },
        'data':{
            'playlists': '/mnt/bulk/recsys18/playlist_hash_ss.csv',
            'tracks': '/mnt/bulk/recsys18/track_hash_ss.csv',
            'train': '/mnt/bulk/recsys18/playlist_track_ss_train.csv',
            'test': '/mnt/bulk/recsys18/playlist_track_ss_test.csv',
            'artist2track': '/mnt/bulk/recsys18/artist_track_ss.csv',
        },
        'model_out': './models/',
        'log_out': './logs/'
    },

    'hyper_parameters':{
        'eval_while_fit': True,
        'sample_weight': False,
        'sample_weight_power': 3./4,
        'sample_threshold': 1e-6,
        'with_context': True,
        'num_epochs': 100,
        'neg_sample': 5,
        'optimizer': optim.Adagrad,
        'learn_rate': 0.1,
        'batch_size': 64,
        'n_embedding': 200,
        'n_out_embedding': 128,
        'n_hid': 128,
        'learn_metric': False,
        'non_lin': nn.ReLU,
        'dropout': False,
        'l2': 1e-4,
        'alpha': 0,
        'ngram_n': 3
    },

    'evaluation':{
        'cutoff':500
    }
}

