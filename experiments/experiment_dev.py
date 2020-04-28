"""Experiment on the development set."""
import json

from cat.simple import get_scores, rbf_attention
from cat.dataset import restaurants_train
from reach import Reach
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict, Counter


GAMMA = .03
BEST_ATT = {"n_noun": 980}
BEST_RBF = {"n_noun": 200}

if __name__ == "__main__":

    scores = defaultdict(dict)
    r = Reach.load("embeddings/restaurant_vecs_w2v.vec",
                   unk_word="<UNK>")

    att = rbf_attention
    datums = list(restaurants_train())

    d = json.load(open("data/nouns_restaurant.json"))
    nouns = Counter()
    for k, v in d.items():
        if k.lower() in r.items:
            nouns[k.lower()] += v

    if att == rbf_attention:
        r.vectors[r.items["<UNK>"]] = r.vectors.max()

    if att == rbf_attention:
        candidates, _ = zip(*nouns.most_common(BEST_RBF["n_noun"]))
    else:
        candidates, _ = zip(*nouns.most_common(BEST_ATT["n_noun"]))

    aspects = [[x] for x in candidates]

    for idx, (instances, y, label_set) in enumerate(datums):

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
        scores[idx] = (f1_score, f1_macro)
