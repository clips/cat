"""Creating fragments takes a long time so we treat it as a
pre-processing step."""
import logging
import json

from gensim.models import Word2Vec
from cat.fragments import create_noun_counts
from cat.utils import conll2text
from collections import Counter

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    paths = ["data/my_data.conllu"]
    create_noun_counts(paths,
                       "data/nouns.json")
    conll2text(paths, "data/all_txt.txt")
    corpus = [x.lower().strip().split()
              for x in open("data/all_txt.txt")]

    f = Word2Vec(corpus,
                 sg=0,
                 negative=5,
                 window=10,
                 size=200,
                 min_count=2,
                 iter=5,
                 workers=10)

    f.wv.save_word2vec_format("embeddings/my_word_vectors.vec")

    d = json.load(open("data/nouns.json"))
    nouns = Counter()
    for k, v in d.items():
        if k.lower() in f.wv.items:
            nouns[k.lower()] += v

    nouns, _ = zip(*sorted(nouns.items(),
                           key=lambda x: x[1],
                           reverse=True))

    json.dump(nouns, open("data/aspect_words.json", "w"))
