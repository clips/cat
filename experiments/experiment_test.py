"""Experiment on the test data."""
import json
import numpy as np

from cat.simple import get_scores, attention, rbf_attention
from cat.dataset import restaurants_test
from reach import Reach
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict, Counter
from itertools import product


GAMMA = .03
BEST_ATT = {"n_noun": 980}
BEST_RBF = {"n_noun": 200}

if __name__ == "__main__":

    scores = defaultdict(dict)
    r = Reach.load("embeddings/restaurant_vecs_w2v.vec",
                   unk_word="<UNK>")

    d = json.load(open("data/nouns_restaurant.json"))

    nouns = Counter()
    for k, v in d.items():
        if k.lower() in r.items:
            nouns[k.lower()] += v

    embedding_paths = ["embeddings/restaurant_vecs_w2v.vec"]
    bundles = ((rbf_attention, attention), embedding_paths)

    for att, path in product(*bundles):
        r = Reach.load(path, unk_word="<UNK>")

        if att == rbf_attention:
            candidates, _ = zip(*nouns.most_common(BEST_RBF["n_noun"]))
        else:
            candidates, _ = zip(*nouns.most_common(BEST_ATT["n_noun"]))

        aspects = [[x] for x in candidates]

        for idx, (instances, y, label_set) in enumerate(restaurants_test()):

            s = get_scores(instances,
                           aspects,
                           r,
                           label_set,
                           gamma=GAMMA,
                           remove_oov=False,
                           attention_func=att)

            y_pred = s.argmax(1)
            f1_score = precision_recall_fscore_support(y, y_pred)
            f1_macro = precision_recall_fscore_support(y,
                                                       y_pred,
                                                       average="weighted")
            scores[(att, path)][idx] = (f1_score, f1_macro)

    att_score = {k: v for k, v in scores.items() if k[0] == attention}
    att_per_class = [[z[x][0][:-1] for x in range(3)]
                     for z in att_score.values()]
    att_per_class = np.stack(att_per_class).mean(0)
    att_macro = np.mean([v[2][1][:-1] for v in att_score.values()], 0)

    rbf_score = {k: v for k, v in scores.items() if k[0] == rbf_attention}
    rbf_per_class = [[z[x][0][:-1] for x in range(3)]
                     for z in rbf_score.values()]
    rbf_per_class = np.stack(rbf_per_class).mean(0)
    rbf_macro = np.mean([v[2][1][:-1] for v in rbf_score.values()], 0)
