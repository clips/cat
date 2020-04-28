"""Grid search over the development data."""
import numpy as np
import pandas as pd
import json

from cat.simple import (get_scores,
                        attention,
                        rbf_attention)
from cat.dataset import restaurants_train
from reach import Reach
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict, Counter
from itertools import product
from tqdm import tqdm


if __name__ == "__main__":

    scores = defaultdict(dict)

    r = Reach.load("embeddings/restaurant_vecs_w2v.vec",
                   unk_word="<UNK>")

    nouns = json.load(open("data/nouns_restaurant.json"))

    c = Counter()
    for k, v in nouns.items():
        if k in r.items:
            c[k.lower()] += (v)

    aspects, _ = zip(*c.most_common(1000))
    aspects = [[x] for x in aspects]

    gamma = np.arange(.0, .1, .01)
    attentions = [(-1, attention)]
    attentions.extend(product(gamma, [rbf_attention]))
    noun_cands = np.arange(10, 1000, 10)

    fun2name = {attention: "att", rbf_attention: "rbf"}

    pbar = tqdm(total=(len(attentions) *
                       len(noun_cands)))

    df = []
    datas = list(restaurants_train())

    for g, att_func in attentions:
        if att_func == rbf_attention:
            r.vectors[r.items["<UNK>"]] += 10e5
        else:
            r.vectors[r.items["<UNK>"]] *= 0

        for n_noun in noun_cands:
            a = aspects[:n_noun]
            for idx, (inst,
                      y,
                      label_set) in enumerate(datas):

                s = get_scores(inst,
                               a,
                               r,
                               label_set,
                               gamma=g,
                               remove_oov=False,
                               attention_func=att_func)

                y_pred = s.argmax(1)
                f1_macro = precision_recall_fscore_support(y,
                                                           y_pred,
                                                           average="weighted")[:-1]  # noqa
                row = (g,
                       fun2name[att_func],
                       n_noun,
                       idx,
                       *f1_macro)
                df.append(row)
            pbar.update(1)
    df = pd.DataFrame(df, columns=("gamma",
                                   "function",
                                   "n_noun",
                                   "dataset",
                                   "p",
                                   "r",
                                   "f1"))
    df.to_csv("results_grid_search.csv")
