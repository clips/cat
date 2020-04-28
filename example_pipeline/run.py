import json

from cat.simple import get_scores, rbf_attention
from reach import Reach
from collections import defaultdict


GAMMA = .03
N_ASPECT_WORDS = 200

if __name__ == "__main__":

    scores = defaultdict(dict)
    r = Reach.load("embeddings/my_word_vectors.vec",
                   unk_word="<UNK>")

    aspects = [[x] for x in json.load(open("data/aspect_words.json"))]
    aspects = aspects[:N_ASPECT_WORDS]

    instances = ["text_1".split(), "text_2".split()]
    label_set = {"label1", "label2", "label3"}

    s = get_scores(instances,
                   aspects,
                   r,
                   label_set,
                   gamma=GAMMA,
                   remove_oov=False,
                   attention_func=rbf_attention)

    pred = s.argmax(1)
