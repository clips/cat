"""Get nouns from conllu files"""
import pyconll
import re
import json

from tqdm import tqdm
from copy import copy
from collections import Counter, defaultdict
from itertools import chain


ARROW = re.compile(r"(<-|->)")


def get_fragments(path, from_pos, to_pos, max_length):
    """Get all fragments from every sentence in a file."""
    all_tokens = [x for x in trees_from_conll(path) if x]
    result = defaultdict(list)
    for id, tokens in all_tokens:
        result[id].extend(search(tokens, from_pos, to_pos, max_length))
    return result


def trees_from_conll(path):
    """Get all trees for every sentence in a conll file."""
    for x in pyconll.iter_from_file(path):
        yield tree(x)


def tree(s):
    """Preprocess a tree to a dict."""
    tokens = {t.id: {"text": t.form.lower(),
                     "pos": t.upos,
                     "id": t.id}
              for t in s}
    if not tokens:
        return s.id.split(".")[0], []
    for token in s:
        idx = token.id
        try:
            # ROOT has a head of None
            nb = token.head
        except (ValueError, TypeError) as e:
            print(e, [x.form for x in s])
        try:
            if nb != "0":
                tokens[idx][f"<-{token.deprel}<-"] = tokens[nb]
                tokens[nb][f"->{token.deprel}->"] = tokens[idx]
        except KeyError as e:
            print(e, tokens)
            return s.id.split(".")[0], []

    return s.id.split(".")[0], list(zip(*sorted(tokens.items())))[1]


def search(tokens, from_pos, to_pos, max_length):
    """
    Search for all patterns starting with POS tag 'f' of max_length.

    Parameters
    ----------
    tokens : list of dict
        A list of dictionaries.
    from_pos : string
        The POS tag to search from.
    to_pos : string
        The POS tag to search to.
    max_length : int
        the maximum length in dependencies to search for.

    Returns
    -------
    result : list
        A list of (word, pattern, word) triples.

    """
    # start the search with tokens with the correct POS.
    result = []
    for token in [t for t in tokens if t["pos"] == from_pos]:
        # return all candidates.
        r = []
        for x in list(_search(token, to_pos, 0, max_length, [], set())):
            pos, text = zip(*x)
            pos_string = "".join(pos)
            pos = ARROW.split(pos_string)
            c = Counter(pos)
            if c[from_pos] > 1 or c[to_pos] > 1:
                continue
            # print(pos, x)
            if pos and pos[0] == from_pos and pos[-1] == to_pos:
                r.append((text[0], pos_string, text[-1]))
        if r:
            result.append(sorted(r, key=lambda x: x[1])[0])

    return result


def _search(token, to, length, max_length, path, visited, dep=None):
    """Recursive function for searching trees."""
    p = copy(path)
    if dep is None:
        p.append([f"{token['pos']}", token["text"]])
    else:
        p.append([f"{dep}{token['pos']}", token["text"]])

    visited.add(token["id"])
    paths = [p]
    if length < max_length:
        for k, v in [(k, v) for k, v in token.items()
                     if k not in {"id", "pos", "text"}]:
            if v["id"] in visited:
                continue
            paths.extend(_search(v, to, length+1, max_length, p, visited, k))
    return [x for x in paths if x]


def create_fragments(in_files, out_path, max_length):
    """Create fragments from all conllu files in a folder."""
    fragments = defaultdict(list)
    for path in tqdm(in_files):
        for k, v in get_fragments(path, "ADJ", "NOUN", max_length).items():
            fragments[k].extend(v)

    json.dump(fragments, open(out_path, 'w'))


def load_fragments(path_to_json,
                   max_path_length=5,
                   words=None):
    """
    Loads fragments from a JSON file.

    Parameters
    ----------
    path_to_json : str
        The path to the json file extracted by create_fragments

    words : iterable
        An iterable of words. Only words in this set are kept.

    max_path_length : int, default 5
        The maximum path length to extract

    Returns
    -------
    fragments : tuple of triples
        A tuple consisting of (adjective, construction, noun) triples.

    """
    fragments = json.load(open(path_to_json))
    _, fragments = zip(*fragments.items())
    fragments = list(chain(*fragments))

    num_arrows = (max_path_length * 4) + 1
    fragments = [x for x in fragments
                 if len(ARROW.split(x[1])) <= num_arrows]

    if words:
        fragments = [(x, y, z) for x, y, z in fragments if
                     x in words and z in words]

    return fragments


def create_noun_counts(in_files, out_path):
    """Get all noun counts."""
    c = Counter()
    for path in in_files:
        c.update(nouns_from_conll(path))

    json.dump(dict(c), open(out_path, 'w'))


def nouns_from_conll(path):
    """Get all nouns, regardless of adjectival modification."""
    for sent in pyconll.iter_from_file(path):
        for token in sent:
            if token.upos == "NOUN":
                yield token.form.lower()
