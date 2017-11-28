#!/usr/bin/env python3
import os

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from embedding_evaluation.load_my_embedding import load_embedding_textfile


def process_mcrae(vocab=None):
    data_path = os.environ["EMBEDDING_EVALUATION_DATA_PATH"]
    datasets_path = os.path.join(data_path, "mcrae", "caracteristics.txt")
    all_words_path = os.path.join(data_path, "mcrae", "all_words.txt")

    with open(all_words_path, "r") as f:
        all_words = f.read().splitlines()
    keep_word = [True if word in set(vocab) else False for word in all_words]
    all_words = [word for i, word in enumerate(all_words) if keep_word[i]] # filter out words that are not in vocab

    datasets = {}
    with open(datasets_path, "r") as f:
        lines = f.read().splitlines()
    for l in lines:
        temp = l.split(",")
        if temp[1] not in datasets.keys():
            datasets[temp[1]] = {}
        labels = [lab for i, lab in enumerate(list(map(float, temp[2:]))) if keep_word[i]]
        datasets[temp[1]][temp[0]] = np.array(labels)

    return datasets, all_words


def evaluate_one_dataset(labels, features, rerun=2):
    scores = []
    for i in range(rerun):
        clf = LinearSVC(class_weight="balanced")
        score = cross_val_score(clf, features, labels, cv=5, scoring="f1", n_jobs=1)
        scores.extend(score)

    return scores

class EvaluationFeatureNorm:

    def __init__(self, entity_subset=None, vocab_path=None, vocab=None):
        assert(vocab_path is not None or vocab is not None) # give vocab_path or vocab to evaluation_mcrae
        self.vocab = vocab
        if vocab_path is not None:
            with open(vocab_path, "r") as f:
                self.vocab = f.read().splitlines()
        self.datasets, self.all_words = process_mcrae(self.vocab)

    def evaluate(self, my_embedding):
        features = np.array([my_embedding[word] for word in self.all_words])

        results = {}
        for category, sub_dataset in self.datasets.items():
            scores = []
            for caracteristic, labels in sub_dataset.items():
                score = evaluate_one_dataset(labels, features)
                scores.extend(score)

            mean = np.mean(scores)
            std = np.std(scores)

            results[category] = {"mean": mean, "std": std}
        return results

