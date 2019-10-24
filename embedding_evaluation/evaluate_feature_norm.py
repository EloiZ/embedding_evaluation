#!/usr/bin/env python3
import os

import numpy as np
import warnings
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import f1_score

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

def evaluate_one_dataset(labels, features, vocab_list):
    scores = []
    predicted_score = []
    ground_truth = []
    word_list = []

    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    kf.get_n_splits()

    # manual splits
    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        words = vocab_list[test_index]

        clf = LinearSVC(class_weight="balanced")
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        score = f1_score(y_test, y_pred)
        scores.append(score)

        predicted_score.extend(list(y_pred))
        ground_truth.extend(list(y_test))
        word_list.extend(words)

    return scores, predicted_score, ground_truth, word_list


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

        all_words_array = np.array(all_words)

        results = {}
        for category, sub_dataset in datasets.items():
            scores = []
            results[category] = {"caracteristic": {}}

            for caracteristic, labels in sub_dataset.items():
                score, predicted_score, ground_truth, word_list = evaluate_one_dataset(labels, features, all_words_array)
                scores.extend(score)

                # Fine grain results for words
                results[category]["caracteristic"][caracteristic] = {}
                results[category]["caracteristic"][caracteristic]["predicted_score"] = predicted_score
                results[category]["caracteristic"][caracteristic]["ground_truth"] = ground_truth
                results[category]["caracteristic"][caracteristic]["word_list"] = word_list

            mean = np.mean(scores)
            std = np.std(scores)

            results[category]["mean"] = mean
            results[category]["std"] = std
            results[category]['N'] = len(sub_dataset)

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

