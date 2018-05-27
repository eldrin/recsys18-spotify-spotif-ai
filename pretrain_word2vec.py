import os
import re
import random
from random import shuffle
from itertools import chain
import cPickle as pkl

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from nltk.tokenize.nist import NISTTokenizer
# from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

from prefetch_generator import background
from tqdm import tqdm, trange


def load_n_process_data(fn, title_column=1, sep='\t', context_win=2):
    """"""
    # load raw data
    with open(fn) as f:
        d = pd.DataFrame(
            [ll.replace('\n', '').split('\t') for ll in f.readlines()]
        )

    # get uniq words
    nist = NISTTokenizer()
    sentences = d[1].apply(
        lambda a: nist.international_tokenize(
            re.sub(r'\([^)]*\)', '', a).decode('utf-8'),
            lowercase=True
        )
    )
    # simple remedy
    missing = sentences.apply(len) == 0
    sentences[missing] = d[missing][1].apply(lambda x: x.lower().split())

    words = set(list(chain.from_iterable(sentences)))
    words_hash = {v: k for k, v in enumerate(words)}

    # make input-out pairs
    data = []
    win_sz = 2 * context_win + 1
    for i, cont in tqdm(sentences.iteritems(), total=sentences.shape[0], ncols=80):
        if len(cont) > 1:
            windowed = [cont[i: i + win_sz] for i in range(len(cont) - win_sz + 1)]
            for c in windowed:
                x = c[2]
                for c_ in c:
                    if x != c_:
                        data.append((x, c_))

    return sentences, data, words_hash


@background(max_prefetch=100)
def sample_generator(data, words_hash, batch_size):
    """"""
    shuffle(data)
    batch_x, batch_y = [], []
    n_words = len(words_hash)
    for x, y in tqdm(data, total=len(data), ncols=80):
        batch_x.append(words_hash[x])
        batch_y.append(words_hash[y])
        if len(batch_x) >= batch_size:
            yield (
                Variable(torch.cuda.LongTensor(batch_x)),
                Variable(torch.cuda.LongTensor(batch_y))
            )
            batch_x = []
            batch_y = []


class SkipGram(nn.Module):
    """"""
    def __init__(self, n_components, n_words):
        """"""
        super(SkipGram, self).__init__()
        self.emb = nn.Embedding(n_words, n_components)
        self.emb.weight.requires_grad = True
        self.linear = nn.Linear(n_components, n_words)

    def forward(self, w):
        """"""
        h = self.emb(w)
        o = self.linear(h)
        return o  # prob. over all words (HEAVY)


if __name__ == "__main__":

    h = 300
    bs = 256
    n_epoch = 100
    l2 = 1e-4
    lr = .001
    adam_eps = 1e-5
    train_ratio = 0.9
    report_every = 200
    use_gensim = True

    # preparing data
    track_fn = './data/track_hash_ss.csv'
    sentences, data, words_hash = load_n_process_data(track_fn)

    if use_gensim:
        model = Word2Vec(
            sentences, size=h, window=2, min_count=1, workers=8, sg=1,
            hs=0, negative=15, iter=20
        )

        # convert gensim model to ndarray
        embedding_matrix = np.zeros((len(model.wv.vocab), h))
        print len(model.wv.vocab)
        print len(model.wv.index2word)
        for i in range(len(model.wv.vocab)):
            emb = model.wv[model.wv.index2word[i]]
            if emb is not None:
                embedding_matrix[i] = emb
        # save relavant data
        np.save('./data/w_emb_skipgram_track_ttl_gensim.npy', embedding_matrix)
        pkl.dump(model.wv.index2word, open('./data/track_id2word.pkl', 'wb'))

    else:
        # train / test split
        shuffle(data)
        train_bound = int(len(data) * train_ratio)
        train_data = data[:train_bound]
        test_data = data[train_bound:]
        print('num of samples: {:d}'.format(len(data)))
        print('num of uniq words: {:d}'.format(len(words_hash)))

        # building model
        model = SkipGram(h, len(words_hash)).cuda()

        # set loss / optimizer
        f_loss = nn.CrossEntropyLoss().cuda()
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            weight_decay=l2, lr=lr, eps=adam_eps)

        # main training loop
        model.train()
        try:
            k = 0  # num of update
            epoch = trange(n_epoch, ncols=80)
            for n in epoch:
                for x, y in sample_generator(train_data, words_hash, bs):
                    # flush grad
                    opt.zero_grad()

                    # forward pass
                    y_pred = model.forward(x)

                    # calc loss
                    l = f_loss(y_pred, y)

                    # back-propagation
                    l.backward()

                    # update
                    opt.step()
                    k += 1

                    if k % report_every == 0:
                        model.eval()  # temporarily switch off to evaluation mode
                        # fetch some test samples
                        test_batch = random.sample(test_data, 128)
                        Xv = Variable(torch.cuda.LongTensor(map(lambda x: words_hash[x[0]], test_batch)))
                        true_v = np.array(map(lambda x: words_hash[x[1]], test_batch))
                        pred_v = np.argsort(model.forward(Xv).data.cpu().numpy(), axis=1)[:,-1]
                        acc = accuracy_score(true_v, pred_v)

                        # log
                        epoch.set_description(
                            '[tloss: {:.3f} / vacc: {:.3f}]'.format(float(l.data), acc)
                        )
                        model.train()

        except KeyboardInterrupt:
            print('[Warning] User stopped the training!')
        # switch off to evaluation mode
        model.eval()

        # save embedding
        np.save('./data/w_emb_skipgram_track_ttl.npy', model.emb.weight.data.cpu().numpy())
