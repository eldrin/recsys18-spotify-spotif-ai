from torch import nn

CONFIG = {
    'path':{
        'embeddings':{
            'U': './data/bpr_U.npy',
            'V': './data/bpr_V.npy',
            'W': './data/bpr_W.npy',
            'X': './data/spotify_feature_popularity_scaled_ss2.npy'
        },
        'data':{
            'train': './data/playlist_track_ss_train.csv',
            'test': './data/playlist_track_ss_test.csv',
            'artist2track': './data/artist_track_ss.csv',
        },
        'model_out': './models/',
        'log_out': './logs/'
    },

    'hyper_parameters':{
        'eval_while_fit': True,
        'sample_weight': False,
        'sample_weight_power': 3./4,
        'sample_threshold': 1e-6,
        'num_epochs': 100,
        'neg_sample': 10,
        'optimizer': nn.Adagrad,
        'learn_rate': 0.001,
        'batch_size': 128,
        'mlp_arch': [],
        'learn_metric': False,
        'non_lin': nn.ReLU,
        'dropout': False,
        'l2': 1e-8,
        'alpha': 0
    },

    'evaluation':{
        'cutoff':500
    }
}

