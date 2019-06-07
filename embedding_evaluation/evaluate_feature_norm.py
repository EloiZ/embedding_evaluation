#!/usr/bin/env python3
import os

import numpy as np
import warnings
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from embedding_evaluation.load_embedding import load_embedding_textfile


def process_mcrae(vocab=None):
    data_path = os.environ["EMBEDDING_EVALUATION_DATA_PATH"]
    datasets_path = os.path.join(data_path, "mcrae", "caracteristics.txt")
    all_words_path = os.path.join(data_path, "mcrae", "all_words.txt")

    with open(all_words_path, "r") as f:
        all_words = f.read().splitlines()
    if vocab is not None:
        vocab_set = set(vocab)
        keep_word = [word in vocab_set for word in all_words]
        all_words = [word for i, word in enumerate(all_words) if keep_word[i]] # filter out words that are not in vocab
    else:
        keep_word = [True for word in all_words]

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
        #assert(vocab_path is not None or vocab is not None) # give vocab_path or vocab to evaluation_mcrae
        self.vocab = vocab
        if vocab_path is not None:
            with open(vocab_path, "r") as f:
                self.vocab = f.read().splitlines()
        self.datasets, self.all_words = process_mcrae(self.vocab)

    def words_in_benchmarks(self):
        return set(self.all_words)

    def filter_dataset(self, embeddings):
        '''Discard features and labels of words with no embedding.'''
        vocab = set(embeddings.keys())
        keep_word = [True if word in vocab else False for word in self.all_words]
        keep_n = keep_word.count(True)
        if keep_n == len(self.all_words):
            return self.datasets, self.all_words
        datasets = {}
        for category, sub_dataset in self.datasets.items():
            datasets.setdefault(category, {})
            characteristics = {}
            scores = []
            for characteristic, labels in sub_dataset.items():
                new_labels = [lab for i, lab in enumerate(labels) if keep_word[i]]
                # if new_labels are all '1' or all '0', discard the characteristic.
                # Note: it does not happen!
                if all(new_labels) or not any(new_labels):
                    warnings.warn("Invalid characteristic {}/{}".format(category, characteristic))
                    continue
                characteristics[characteristic] = np.array(new_labels)
            if len(characteristics):
                datasets[category] = characteristics
        return datasets, [word for i, word in enumerate(self.all_words) if keep_word[i]]

    def evaluate(self, my_embedding):
        datasets, all_words = self.filter_dataset(my_embedding)
        features = np.array([my_embedding[word] for word in all_words])
        results = {}
        for category, sub_dataset in datasets.items():
            scores = []
            for caracteristic, labels in sub_dataset.items():
                score = evaluate_one_dataset(labels, features)
                scores.extend(score)

            mean = np.mean(scores)
            std = np.std(scores)

            results[category] = {"mean": mean, "std": std, 'N' : len(sub_dataset)}
        # Note: macro and mcro is the same (same number of instances in each category)
        macro_mean = 0
        micro_mean = 0
        total_n = 0
        for cat, r in results.items():
            macro_mean += r['mean']
            micro_mean += r['mean'] * r['N']
            total_n += r['N']
        macro_mean /= len(results)
        micro_mean /= total_n
        return  {'macro' : macro_mean,
                 'micro' : micro_mean,
                 "total_words" : len(self.all_words),
                 "words_with_embedding" : len(all_words),
                 'results' : results}

