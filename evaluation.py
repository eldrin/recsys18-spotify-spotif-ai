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
