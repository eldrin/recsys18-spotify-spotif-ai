from functools import partial
from torch import nn
from torch import optim

CONFIG = {
    'path':{
        'embeddings':{
            'U': '/mnt/bulk/recsys18/models/wrmf_U_ss.npy',
            'V': '/mnt/bulk/recsys18/models/wrmf_V_ss.npy',
            'W': '/mnt/bulk/recsys18/models/wrmf_W_ss.npy',
            'X': '/mnt/bulk/recsys18/10percent/track_audio_feature_ss.npy',
            'track_word': './data/w_emb_skipgram_track_ttl_gensim.npy',
            'track_id2word': './data/track_id2word.pkl'
        },
        'data':{
            'playlists': '/mnt/bulk/recsys18/playlist_hash_ss.csv',
            'tracks': '/mnt/bulk/recsys18/track_hash_ss.csv',
            'artists': '/mnt/bulk/recsys18/artist_hash_ss.csv',
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
        'track_emb': 'artist_ngram',
        'loss': 'SGNS',  # {'SGNS', 'MSE'}
        'optimizer': optim.Adam,
        'learn_rate': 0.01,
        'batch_size': 256,
        'mlp_arch': [],
        'n_embedding': 128,
        'n_out_embedding': 50,
        'n_hid': 256,
        'n_layers': 2,
        'learn_metric': False,
        'non_lin': nn.ReLU,
        'dropout': False,
        'l2': 1e-7,
        'alpha': 0,
        'ngram_n': 3
    },

    'evaluation':{
        'cutoff':500
    }
}

