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

def evaluate_one_benchmark(my_embedding, embedding_fallback, benchmark, entity_subset=None, several_embeddings=False):
    """
    my_embedding: {word: np.array}
    entity_subset: [ent] a list of entities on which you want to evaluate the benchmark separately
    benchmark: {(word1, word2): score}
    """
    gold_list = []
    target_list = []
    final_word_pair_list = []
    gold_list_ent_only = []
    target_list_ent_only = []
    count_used_pairs = 0
    for (word1, word2), gold_score in benchmark.items():
        if word1 not in my_embedding or word2 not in my_embedding:
            # see whether we can use fallback embeddings
            if embedding_fallback is None or word1 not in embedding_fallback or word2 not in embedding_fallback:
                continue
            v1 = embedding_fallback[word1]
            v2 = embedding_fallback[word2]
        else:
            v1 = my_embedding[word1]
            v2 = my_embedding[word2]

        count_used_pairs += 1
        if several_embeddings:
            sim = average_cosine_similarity(v1, v2)
        else:
            sim = cosine_similarity(v1, v2)

        final_word_pair_list.append((word1, word2))
        gold_list.append(gold_score)
        target_list.append(sim)

        if entity_subset is not None:
            if word1 in entity_subset and word2 in entity_subset:
                gold_list_ent_only.append(gold_score)
                target_list_ent_only.append(sim)
    #used_pairs = count_used_pairs / len(benchmark)

    sp_all, _ = spearmanr(target_list, gold_list)
    sp_ent_only = 0

    if entity_subset is not None:
        sp_ent_only, _ = spearmanr(target_list_ent_only, gold_list_ent_only)
        result["entity_subset"] = sp_ent_only

    return {"all_entities": sp_all,
            "entity_subset": sp_ent_only,
            'total_pairs': len(benchmark),
            'used_pairs' : count_used_pairs,
            "gold_list": list(map(str, gold_list)),
            "target_list": list(map(str, target_list)),
            "final_word_pair_list": final_word_pair_list,
            }

class EvaluationSimilarity:

    def __init__(self, entity_subset=None, benchmark_subset=False):
        self.all_benchmarks = process_benchmarks(benchmark_subset=benchmark_subset)
        self.entity_subset = entity_subset

    def words_in_benchmarks(self, bname=None):
        if bname == None:
            bname = set(self.all_benchmarks.keys())
        vocab = set()
        for benchmark_string in bname:
            benchmark = self.all_benchmarks[benchmark_string]
            for (word1, word2), gold_score in benchmark.items():
                vocab.add(word1)
                vocab.add(word2)
        return vocab

    def evaluate(self, my_embedding, embeding_fallback = None, several_embeddings=False):
        results = {}
        for benchmark_string, benchmark in self.all_benchmarks.items():
            results[benchmark_string] = evaluate_one_benchmark(my_embedding, embeding_fallback, benchmark, self.entity_subset, several_embeddings=several_embeddings)
        return results

