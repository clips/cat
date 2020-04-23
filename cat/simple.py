"""Simple method."""
import numpy as np
from collections import defaultdict
from .utils import normalize
from sklearn.metrics.pairwise import rbf_kernel
from collections import Counter


def get_aspects(fragments, embeddings, n_adj_seed, n_nouns, min_count):
    """Get aspects based on fragments."""
    adj, _, noun = zip(*fragments)
    adj_cand, _ = zip(*Counter(adj).most_common(n_adj_seed))

    cands = candidate(embeddings,
                      adj,
                      noun,
                      adj_cand,
                      n_nouns,
                      min_count)

    return cands


def candidate(embeddings,
              adj,
              noun,
              seed_words,
              n_nouns,
              min_count):
    """
    Generates candidate aspects based on adjective co-occurrences

    Parameters
    ----------
    embeddings : Reach
        A Reach instance containing the word embeddings.
    constructions : list of tuples
        A list of adjective noun tuples.
    seed_words : list of str
        A list of strings. All these words should be in vocab for the
        given embeddings model.
    frequency_threshold : int
        Any noun occurring fewer times than this threshold is discarded
    n_nouns : int
        The amount of items to return

    Returns
    -------
    candidates : dict
        A dictionary mapping strings to their scores.

    """
    a = list(set(adj))
    sims = embeddings.similarity(a, seed_words).max(1)
    adj_scores = dict(zip(a, sims))

    noun_scores = defaultdict(lambda: [0, 0])
    for adj, noun in zip(adj, noun):
        noun_scores[noun][0] += adj_scores[adj]
        noun_scores[noun][1] += 1

    noun_scores = {k: v[0] for k, v in noun_scores.items()
                   if v[1] > min_count}

    return sorted(noun_scores.items(), key=lambda x: x[1])[-n_nouns:]


def rbf_attention(vec, memory, gamma, **kwargs):
    """
    Single-head attention using RBF kernel.

    Parameters
    ----------
    vec : np.array
        an (N, D)-shaped array, representing the tokens of an instance.
    memory : np.array
        an (M, D)-shaped array, representing the memory items
    gamma : float
        the gamma of the RBF kernel.

    Returns
    -------
    attention : np.array
        A (1, N)-shaped array, representing a single-headed attention mechanism

    """
    z = rbf_kernel(vec, memory, gamma)
    s = z.sum()
    if s == 0:
        # If s happens to be 0, back off to uniform
        return np.ones((1, len(vec))) / len(vec)
    return (z.sum(1) / s)[None, :]


def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis, keepdims=True))
    s = e_x.sum(axis=axis, keepdims=True)
    return e_x / s


def attention(vec, memory, **kwargs):
    """
    Standard multi-head attention mechanism.

    Parameters
    ----------
    vec : np.array
        an (N, D)-shaped array, representing the tokens of an instance.
    memory : np.array
        an (M, D)-shaped array, representing the memory items

    Returns
    -------
    attention : np.array
        A (M, N)-shaped array, representing the attention over all memories.

    """
    z = memory.dot(vec.T)
    return softmax(z)


def mean(vec, aspect_vecs, **kwargs):
    """Just a mean weighting."""
    return (np.ones(len(vec)) / len(vec))[None, :]


def get_scores(instances,
               aspects,
               r,
               labels,
               remove_oov=False,
               attention_func=attention,
               **kwargs):
    """Scoring function."""
    assert all([x in r.items for x in labels])
    label_vecs = normalize(r.vectorize(labels))
    aspect_vecs = [x.mean(0)
                   for x in r.transform(aspects,
                                        remove_oov=False)]
    aspect_vecs = np.stack(aspect_vecs)
    if len(instances) == 1:
        instances = [instances]

    t = r.transform(instances, remove_oov=remove_oov)

    out = []
    for vec in t:
        att = attention_func(vec, aspect_vecs, **kwargs)
        # Att = (n_heads, n_words)
        z = att.dot(vec)
        # z = (n_heads, n_dim)
        x = normalize(z).dot(label_vecs.T)
        # x = (n_heads, n_labels)
        out.append(x.sum(0))
    return np.stack(out)
