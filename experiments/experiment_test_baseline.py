import numpy as np

from glob import glob
from cat.simple import get_scores, mean
from cat.dataset import restaurants_test
from reach import Reach
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict


if __name__ == "__main__":

    scores = defaultdict(dict)
    r = Reach.load("embeddings/restaurant_vecs_w2v.vec",
                   unk_word="<UNK>")

    datums = list(restaurants_test())

    for path in sorted(glob("embeddings/restaurant_vecs_w2v.vec")):
        r = Reach.load(path, unk_word="<UNK>")

        for idx, (instances, y, label_set) in enumerate(datums):

            s = get_scores(instances,
                           [["food"]],
                           r,
                           label_set,
                           remove_oov=False,
                           attention_func=mean)

            y_pred = s.argmax(1)
            f1_score = precision_recall_fscore_support(y, y_pred)
            f1_macro = precision_recall_fscore_support(y,
                                                       y_pred,
                                                       average="weighted")
            scores[path][idx] = (f1_score, f1_macro)

    score_per_class = [[z[x][0][:-1] for x in range(3)]
                       for z in scores.values()]
    score_per_class = np.stack(score_per_class).mean(0)

    macro_score = [v[2][1][:-1] for v in scores.values()]
