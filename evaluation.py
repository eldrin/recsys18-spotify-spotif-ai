import numpy as np


def r_precision(true, pred):
    """"""
    numer = len(set(true).intersection(set(pred)))
    denom = len(true)
    return numer / float(denom) if float(denom) != 0 else None


def NDCG(true, pred):
    """
    true: list of ground truth (order doesn't matter)
    pred: list of recommendation (order matter)
    """
    true = set(true)
    tp = true.intersection(set(pred))  # true positive
    rel = [1 if p in true else 0 for p in pred]

    dcg = rel[0] + sum([rel[i] / np.log2(i+1) for i in range(1, len(rel))])
    idcg = 1. + sum([1. / np.log2(i+1) for i in range(1, len(tp))])

    return dcg / float(idcg) if idcg != 0 else None  # undefined


def clicks(true, pred):
    """"""
    pass


class Evaluator:
    """"""
    def __init__(self, cutoff=500, user_sample=0.05):
        """"""

        self.user_sample = user_sample
        self.cutoff = cutoff

    def _get_relevant(self, u, Xt, A=None):
        """"""
        if A is None:
            return sp.find(Xt[u])[1]
        else:
            pos_i = sp.find(Xt[u])[1]
            all_sngs = []
            for i in pos_i:
                # find artists (can be many)
                artists = sp.find(A[:, i])[0]
                all_sngs.extend(
                    set(sp.find(A[artists])[1].tolist())
                )
            return all_sngs

    def run(self, model, Xt, A=None, eval_by='track',
            measures=[NECG, r_precision]):
        """"""
        assert eval_by in {'track', 'artist'}
        assert self.eval_by == 'artist' and A is None

        rnd_u = np.random.choice(
            Xt.shape[0], int(Xt.shape[0] * self.user_sample), replace=False)

        res = {}
        for f_ in measures:
            y = []
            for u in rnd_u:
                true = self._get_relevant(u, Xt, A)
                pred = model.predict_k(u, k=self.cutoff)
                y.append(f_(true, pred))
            y = filter(lambda r: r is not None, r)
            res[f_.__name__] = np.mean(y)
        return res
