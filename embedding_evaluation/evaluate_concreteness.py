#!/usr/bin/env python3
import os

import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import KFold

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

class EvaluationConcreteness:

    def __init__(self, entity_subset=None):
        self.concreteness = process_concreteness()

    def words_in_benchmarks(self):
        vocab = set(self.concreteness.keys())
        return vocab

    def evaluate_one_dataset(self, labels, features, vocab_list):
        scores = []
        predicted_conc = []
        ground_truth_conc = []
        word_list = []
        scores_per_word = []

        kf = KFold(n_splits=5, shuffle=True, random_state=1234)
        kf.get_n_splits()

        # manual splits
        for train_index, test_index in kf.split(features, labels):
            #print(train_index, test_index)
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            words = vocab_list[test_index]

            self.clf = SVR(kernel="rbf", gamma="auto")
            self.clf.fit(X_train, y_train)
            score = self.clf.score(X_test, y_test)
            scores.append(score)

            # Per word score
            y_pred = self.clf.predict(X_test)
            square_res = 0.5 * (y_test - y_pred) ** 2
            #sum_of_squares = (y_test - y_pred.mean()) ** 2

            predicted_conc.extend(list(self.clf.predict(X_test)))
            scores_per_word.extend(list(square_res))
            ground_truth_conc.extend(list(y_test))
            word_list.extend(words)

        return scores, predicted_conc, ground_truth_conc, word_list, scores_per_word


    def evaluate(self, my_embedding):
        labels = []
        features = []
        vocab_list = []
        for word, conc in self.concreteness.items():
            if word in my_embedding:
                labels.append(conc)
                features.append(my_embedding[word])
                vocab_list.append(word)
                #features.append(np.random.randn(300))
        #print("Evaluated words: %d" % len(labels))

        self.features = np.array(features)
        self.labels = np.array(labels)
        self.vocab_list = np.array(vocab_list)
        scores, predicted_conc, ground_truth_conc, word_list, scores_per_word = self.evaluate_one_dataset(self.labels, self.features, self.vocab_list)
        mean = np.mean(scores)
        std = np.std(scores)
        results = {"mean": mean,
                "std": std,
                "total_words": len(self.concreteness),
                "words_with_embedding": len(self.vocab_list),
                "ground_truth": list(map(str, ground_truth_conc)),
                "predicted_conc": list(map(str, predicted_conc)),
                "word_list": word_list,
                "score_per_word": scores_per_word,
                }

        return results

