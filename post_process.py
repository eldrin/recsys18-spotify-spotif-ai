import os
import numpy as np
import pandas as pd

from util import read_data
import fire


def main(mf_user_factors_fn, rnn_user_factors_fn, out_fn, train_fn, test_fn=None):
    """"""
    # load train data
    print('Loading data...')
    y = pd.read_csv(
        train_fn, header=None, index_col=None,
        names=['playlist', 'track', 'value'])

    # check and load test data (if it exists)
    if test_fn is not None:
        if os.path.exists(test_fn):
            yt = pd.read_csv(
                test_fn, header=None, index_col=None,
                names['playlist', 'track', 'value'])
        else:
            raise ValueError('[ERROR] {} is not existing!'.format(test_fn))

    # identify the part where needs to be replaced
    replace_ix = y[y['value'] == 0]['playlist'].unique()

    # load factors
    U_mf = np.load(mf_user_factors_fn)
    U_rnn = np.load(rnn_user_factors_fn)

    # replace!
    print('Processing...')
    U_mf[replace_ix] = U_rnn[replace_ix]

    # save...
    print('Saving...')
    np.save(out_fn, U_mf)


if __name__ == "__main__":
    fire.Fire(main)
