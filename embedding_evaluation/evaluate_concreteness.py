#!/usr/bin/env python3
import os

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

from embedding_evaluation.load_embedding import load_embedding_textfile


def process_concreteness():
    data_path = os.environ["EMBEDDING_EVALUATION_DATA_PATH"]
    dataset_path = os.path.join(data_path, "usf", "concreteness.txt")

    concreteness = {}
    with open(dataset_path, "r") as f:
        data = f.read().splitlines()

    for line in data:
        s = line.split(",")
        concreteness[s[0]] = float(s[1])
    return concreteness


def evaluate_one_dataset(labels, features, rerun=2):
    scores = []
    for i in range(rerun):
        clf = SVR(kernel="rbf")
        score = cross_val_score(clf, features, labels, cv=5)#, scoring="f1")
        #print(score)
        scores.extend(score)
    return scores

class EvaluationConcreteness:

    def __init__(self, entity_subset=None):
        self.concreteness = process_concreteness()

    def evaluate(self, my_embedding):
        labels = []
        features = []
        for word, conc in self.concreteness.items():
            if word in my_embedding:
                labels.append(conc)
                features.append(my_embedding[word])
                #features.append(np.random.randn(300))
        #print("Evaluated words: %d" % len(labels))

        features = np.array(features)
        scores = evaluate_one_dataset(labels, features)
        mean = np.mean(scores)
        std = np.std(scores)
        results = {"mean": mean, "std": std}

        return results["mean"]

