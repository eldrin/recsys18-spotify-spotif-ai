import os
import numpy as np

from util import read_data
from cf import WRMFAttrSim, WRMF, ImplicitALS, TSVD
from evaluation import evaluate

import fire


def main(train_fn, test_fn=None, r=10, attr_fn=None, attr_sim_fn=None,
         gamma=1, epsilon=1, n_epoch=30, cutoff=500,
         beta=1, beta_a=1, beta_b=1, model_out_root=None, model_name='wrmf'):
    """"""
    print('Loading data...')
    d, y = read_data(train_fn)
    if test_fn is not None:
        dt, yt = read_data(test_fn, shape=d.shape)
    a, b = None, None

    if attr_fn is not None:
        a, _ = read_data(attr_fn)
        a_n = a.sum(axis=1)
    if attr_sim_fn is not None:
        b, _ = read_data(attr_sim_fn, shape=(a.shape[0], a.shape[0]))

    print('Fit model (r={:d})!'.format(r))
    if a is not None or b is not None:
        model = WRMFAttrSim(r, beta_a=beta_a, beta_b=beta_b, beta=beta,
                            gamma=gamma, epsilon=epsilon, n_epoch=n_epoch,
                            verbose=True)
        model.fit(d, a, b)
    else:
        if 'wrmf' in model_name:
            model = WRMF(r, beta=beta, gamma=gamma, epsilon=epsilon,
                         n_epoch=n_epoch, verbose=True)
        elif 'implicitals' in model_name:
            model = ImplicitALS(r)
        else:
            raise ValueError(
                '[Error] {} is not supported model!'.format(model_name)
            )
        model.fit(d)

    if test_fn is not None:
        print('Evaluate!')
        res, (trues, preds) = evaluate(model, y, yt, cutoff)
        print(res)

    print('Save Model...')
    if model_out_root is None:
        model_out_root = os.path.join(os.getcwd(), 'data')

    np.save(
        os.path.join(model_out_root, '{}_U.npy'.format(model_name)),
        model.U.astype(np.float32)
    )
    np.save(
        os.path.join(model_out_root, '{}_V.npy'.format(model_name)),
        model.V.astype(np.float32)
    )
    if hasattr(model, 'W'):
        np.save(
            os.path.join(model_out_root, '{}_W.npy'.format(model_name)),
            model.W.astype(np.float32)
        )

if __name__ == "__main__":
    fire.Fire(main)

