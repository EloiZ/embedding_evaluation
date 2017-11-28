#!/usr/bin/env python3
import numpy as np
from scipy.stats import spearmanr

from embedding_evaluation.process_benchmarks import process_benchmarks
from embedding_evaluation.load_embedding import load_embedding_textfile


def cosine_similarity(x, y):
    temp = x / np.linalg.norm(x, ord=2)
    temp2 = y / np.linalg.norm(y, ord=2)
    return np.dot(temp, temp2)

def average_cosine_similarity(x, y):
    """ 5 X 300 ; 5 X 300 """
    temp = x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    temp2 = y / np.linalg.norm(y, ord=2, axis=1, keepdims=True)
    return np.mean(np.dot(temp, temp2.T))

def evaluate_one_benchmark(my_embedding, benchmark, entity_subset=None, several_embeddings=False):
    """
    my_embedding: {word: np.array}
    vocab_to_id: {word: row_in_embedding}
    entity_subset: [ent] a list of entities on which you want to evaluate the benchmark separately
    benchmark: {(word1, word2): score}
    """
    gold_list = []
    target_list = []
    gold_list_ent_only = []
    target_list_ent_only = []
    for (word1, word2), gold_score in benchmark.items():
        if word1 not in my_embedding.keys() or word2 not in my_embedding.keys():
            continue

        if several_embeddings:
            sim = average_cosine_similarity(my_embedding[word1], my_embedding[word2])
        else:
            sim = cosine_similarity(my_embedding[word1], my_embedding[word2])

        gold_list.append(gold_score)
        target_list.append(sim)

        if entity_subset is not None:
            if word1 in entity_subset and word2 in entity_subset:
                gold_list_ent_only.append(gold_score)
                target_list_ent_only.append(sim)

    sp_all, _ = spearmanr(target_list, gold_list)
    sp_ent_only = 0
    if entity_subset is not None:
        sp_ent_only, _ = spearmanr(target_list_ent_only, gold_list_ent_only)

    return {"all_entities": sp_all, "entity_subset": sp_ent_only}


class EvaluationSimilarity:

    def __init__(self, entity_subset=None):
        self.all_benchmarks = process_benchmarks()
        self.entity_subset = entity_subset

    def evaluate(self, my_embedding, several_embeddings=False):
        results = {}
        for benchmark_string, benchmark in self.all_benchmarks.items():
            results[benchmark_string] = evaluate_one_benchmark(my_embedding, benchmark, self.entity_subset, several_embeddings=several_embeddings)
        return results

