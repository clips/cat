"""Simple dataset loader for the 2014, 2015 semeval datasets."""
from sklearn.preprocessing import LabelEncoder
from functools import partial


def loader(instance_path,
           label_path,
           subset_labels,
           split_labels=False,
           mapping=None):

    subset_labels = set(subset_labels)
    labels = open(label_path)
    labels = [x.strip().lower().split() for x in labels]

    instances = []
    for line in open(instance_path):
        instances.append(line.strip().lower().split())

    if split_labels:
        labels = [[l.split("#")[0] for l in x] for x in labels]

    instances, gold = zip(*[(x, y[0]) for x, y in zip(instances, labels)
                            if len(y) == 1 and y[0]
                            in subset_labels])

    if mapping is not None:
        gold = [mapping.get(x, x) for x in gold]

    le = LabelEncoder()
    y = le.fit_transform(gold)
    label_set = le.classes_.tolist()

    return instances, y, label_set


rest_14_test = partial(loader,
                       instance_path="data/restaurant_test_2014_tok.txt",  # noqa
                       label_path="data/labels_restaurant_test_2014.txt",  # noqa
                       subset_labels={"ambience",
                                      "service",
                                      "food"})


rest_14_train = partial(loader,
                        instance_path="data/restaurant_train_2014.txt",  # noqa
                        label_path="data/labels_restaurant_train_2014.txt",  # noqa
                        subset_labels={"ambience",
                                       "service",
                                       "food"})


ganu_test = partial(loader,
                    instance_path="data/test_tok.txt",
                    label_path="data/test_label.txt",
                    subset_labels={"ambience",
                                   "staff",
                                   "food"})


rest_15_train = partial(loader,
                        instance_path="data/restaurant_train_2015_tok.txt",
                        label_path="data/labels_restaurant_train_2015.txt",
                        subset_labels={"ambience",
                                       "service",
                                       "food"},
                        split_labels=True)

rest_15_test = partial(loader,
                       instance_path="data/restaurant_test_2015_tok.txt",
                       label_path="data/labels_restaurant_test_2015.txt",
                       subset_labels={"ambience",
                                      "service",
                                      "food"},
                       split_labels=True)


def restaurants_train():
    yield rest_14_train()
    yield rest_15_train()


def restaurants_test():
    yield rest_14_test()
    yield rest_15_test()
    yield ganu_test()
