import numpy as np
from scipy import sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from util import read_data
from cf import WRMF
from evaluation import r_precision, NDCG

from tqdm import tqdm
import fire


def evaluate(model, dt, cutoff, user_frac=0.05):
    """"""
    # for efficiency, sample 5% of the user to approximate the performance
    rnd_u = np.random.choice(
        dt.shape[0], int(dt.shape[0] * user_frac), replace=False)
    rprec = []
    ndcg = []
    for u in rnd_u:
        true = sp.find(dt[u])[1]
        pred = model.predict_k(u, k=cutoff)

        rprec.append(r_precision(true, pred))
        ndcg.append(NDCG(true, pred))
    rprec = filter(lambda r: r is not None, rprec)
    ndcg = filter(lambda r: r is not None, ndcg)

    out = {
        'NDCG': ndcg,
        'R_Precision': rprec,
    }
    return out


def main(data_fn, attr_fn=None, out_fn=None, cutoff=500, test_ratio=0.7,
         beta=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
         r=[5, 10, 20, 40, 100, 200], n_epoch=10):
    """"""
    print('Loading data...')
    d = read_data(data_fn)
    i, j, v = sp.find(d)
    rnd_idx = np.random.choice(len(i), len(i), replace=False)
    bound = int(len(i) * test_ratio)
    rnd_idx_trn = rnd_idx[:bound]
    rnd_idx_val = rnd_idx[bound:]
    d = sp.coo_matrix((v[rnd_idx_trn], (i[rnd_idx_trn], j[rnd_idx_trn])),
                      shape=d.shape)
    dt = sp.coo_matrix((v[rnd_idx_val], (i[rnd_idx_val], j[rnd_idx_val])),
                       shape=d.shape).tocsr()
    if attr_fn is not None:
        a = read_data(attr_fn)
    else:
        a = None

    result = []
    print('Fit model!')
    for r_ in tqdm(r, ncols=80):
        for b_ in beta:
            model = WRMF(r_, beta=b_, n_epoch=n_epoch)
            model.fit(d, a, dt)

            res = evaluate(model, dt, cutoff)
            result.append({
                'r': r_, 'beta': b_,
                'ndcg': res['NDCG'],
                'rprec': res['R_Precision'],
                'attr': True if attr_fn is not None else False
            })

    if out_fn is not None:
        print('Save result!...')
        pd.DataFrame(result).to_csv(out_fn)


if __name__ == "__main__":
    fire.Fire(main)
