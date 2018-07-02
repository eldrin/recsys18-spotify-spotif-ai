import os
from os.path import isfile
import numpy as np

from util import read_data
from model import ImplicitALS
from evaluation import evaluate

import fire


def main(train_fn, test_fn=None, r=10, alpha=100, n_epoch=15, beta=1e-3,
         cutoff=500, model_out_root=None, model_name='wrmf'):
    """"""
    print('Loading data...')
    d, y = read_data(train_fn)
    if test_fn is not None:
        dt, yt = read_data(test_fn, shape=d.shape)

    print('Fit model (r={:d})!'.format(r))
    model = ImplicitALS(r, beta, alpha, n_epoch)
    model.fit(d)

    if test_fn is not None:
        print('Evaluate!')
        res, (trues, preds) = evaluate(model, y, yt, cutoff)
        print(res)

    if model_out_root is not None:
        if not os.path.exists(model_out_root):
            os.makedirs(model_out_root)

        print('Save Model...')
        if model_out_root is None:
            model_out_root = os.path.join(os.getcwd(), 'data')

        np.save(
            os.path.join(model_out_root, '{}_U.npy'.format(model_name)),
            model.user_factors
        )
        np.save(
            os.path.join(model_out_root, '{}_V.npy'.format(model_name)),
            model.item_factors
        )


if __name__ == "__main__":
    fire.Fire(main)

