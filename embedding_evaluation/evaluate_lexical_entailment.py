#!/usr/bin/env python3
import os

#
# evaluate lexical entailment using the hyperlex dataset
#
# http://people.ds.cam.ac.uk/iv250/hyperlex.html

import numpy as np
from scipy.stats.stats import pearsonr, spearmanr


def cosine_similarity(x, y):
    temp = x / np.linalg.norm(x, ord=2)
    temp2 = y / np.linalg.norm(y, ord=2)
    return np.dot(temp, temp2)


def load_hyperlex_data(filename_path):
    D = []
    with open(filename_path, 'r') as fh:
        fh.readline() # first line is metadata
        for line in fh:
            fields = line.rstrip().split()
            # TODO: should we filter out some relations? (fields[3])
            D.append([fields[0], fields[1], float(fields[4])])
    return D


def process_lexical_entailment():
    data_path = os.environ["EMBEDDING_EVALUATION_DATA_PATH"]
    hyperlex_fname = os.path.join(data_path, "hyperlex", "hyperlex-all.txt")

    lexical_entailment = {}
    lexical_entailment['all'] = load_hyperlex_data(hyperlex_fname)
    return lexical_entailment


def get_word_embedding(word, E, E_fallback=None):
    if word in E:
        return E[word]
    if E_fallback is not None and word in E_fallback:
        return E_fallback[word]
    return None


class EvaluationLexicalEntailment:

    def __init__(self, entity_subset=None):
        self.lexical_entailment = process_lexical_entailment()

    def words_in_benchmarks(self):
        vocab = set()
        for w1, w2, sc in self.lexical_entailment['all']:
            vocab.add(w1)
            vocab.add(w2)
        return vocab

    def evaluate(self, my_embedding, embedding_fallback=None):
        scores = []
        labels = []
        total_pairs = 0
        used_pairs = 0
        for word1, word2, label in self.lexical_entailment['all']:
            total_pairs += 1
            e1 = get_word_embedding(word1, my_embedding, embedding_fallback)
            if e1 is None:
                continue
            e2 = get_word_embedding(word2, my_embedding, embedding_fallback)
            if e2 is None:
                continue
            used_pairs += 1
            scores.append(cosine_similarity(e1, e2))
            labels.append(label)
        rho, _ = pearsonr(scores, labels)
        srho, _ = spearmanr(scores, labels)
        results = {"pearson": rho, "spearman": srho, "total_pairs" : total_pairs, "used_pairs" : used_pairs}
        return results
