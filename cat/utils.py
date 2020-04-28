"""Some utils."""
import pyconll
import numpy as np


def conll2text(paths, outpath):
    """Write a conll file to a text file."""
    with open(outpath, 'w') as f:
        for path in paths:
            for sent in pyconll.iter_from_file(path):
                txt = []
                for x in sent:
                    txt.append(x.form)
                if txt:
                    txt = " ".join(txt).lower()
                    txt = "".join([x for x in txt if x.isprintable()])
                    f.write(f"{txt}\n")


def normalize(x):
    """Normalize a vector while controlling for zero vectors."""
    x = np.copy(x)
    if np.ndim(x) == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / np.linalg.norm(x)
    norm = np.linalg.norm(x, axis=-1)
    mask = norm > 0
    x[mask] /= norm[mask][:, None]
    return x
