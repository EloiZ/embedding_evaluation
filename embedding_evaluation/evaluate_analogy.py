#!/usr/bin/env python3

import os
from sklearn.model_selection import KFold
import numpy as np

class EmbMatrix(object):
    def __init__(self, embeddings):
        # embeddings : { word1 : word2 }
        self.w2idx = {}
        self.idx2w = {}
        self.N = len(embeddings)
        if self.N == 0:
            return
        for i,w in enumerate(embeddings.keys()):
            self.w2idx[w] = i
            self.idx2w[i] = w
        self.dim = 0
        for v in embeddings.values():
            self.dim = len(v)
            break
        matrix = np.ndarray(shape=(self.N, self.dim))
        for w in embeddings.keys():
            matrix[self.w2idx[w]] = embeddings[w]
        # lenght normalize matrix
        norms = np.sqrt(np.sum(matrix**2, axis=1))
        norms[norms == 0] = 1
        self.M = matrix / norms[:, np.newaxis]

    def get_vector(self, word):
        i = self.w2idx.get(word, None)
        if i == None:
            return []
        return self.M[i]

    def get_nn_vector(self, v, topk, exclude_words, norm=False):
        if self.N == 0:
            return []
        if not norm:
            v = v / np.linalg.norm(v, ord=2)
        simv = self.M.dot(v)
        topk = min(topk, len(simv))
        top_indices = np.argsort(simv)[::-1] # trick to reverse array
        # remove words from exclude_words set
        result = []
        for widx in top_indices:
            w = self.idx2w[widx]
            if w in exclude_words:
                continue
            result.append(w)
            if len(result) == topk:
                break
        return result

    def get_nn_word(self, word, topk, exclude_words):
        '''get top-K nearest neighbors of word (excluding words in exclude_words set)'''
        i = self.w2idx.get(word, None)
        if i == None:
            return []
        return get_nn_vector(self.M[i], topk, exclude_words, norm=True)

    def __contains__(self, w):
        return w in self.w2idx;

# From vecto

def get_pair(line):
    if "\t" in line:
        parts = line.lower().split("\t")
    else:
        parts = line.lower().split()
    left = parts[0]
    right = parts[1]
    right = right.strip()
    if "/" in right:
        right = [i.strip() for i in right.split("/")]
    else:
        right = [i.strip() for i in right.split(",")]
    return (left, right)

def process_benchmark_set(dir_path):
    dataset = {}
    for bname in os.listdir(dir_path):
        _ ,ext = os.path.splitext(bname)
        if  ext != '.txt':
            continue
        fname = os.path.join(dir_path, bname)
        for line in open(fname, encoding='utf-8', errors='surrogateescape'):
            if line.strip() == '':
                continue
            (src, tgts) = get_pair(line)
            dataset[src] = tgts[0]
            # if s[1] not in dataset:
            #     dataset[s[1]] = set()
            # dataset[s[1]] = s[0]
    return dataset

def process_benchmark_quads(fname):
    dataset = []
    for line in open(fname, encoding='utf-8', errors='surrogateescape'):
        if line.startswith(':'):
            continue
        qquad = [x.lower() for x in line.strip().split()]
        dataset.append(qquad)
    return dataset

def process_analogy_set():
    data_path = os.environ["EMBEDDING_EVALUATION_DATA_PATH"]
    morf_inflectional_path = os.path.join(data_path, "BATS_3.0", "1_Inflectional_morphology")
    morf_derrivational_path = os.path.join(data_path, "BATS_3.0", "2_Derivational_morphology")
    sem_enciclopedic_path = os.path.join(data_path, "BATS_3.0", "3_Encyclopedic_semantics")
    sem_lexicographic_path = os.path.join(data_path, "BATS_3.0", "4_Lexicographic_semantics")

    analogy = {
        'BATS.morf_inflectional' : process_benchmark_set(morf_inflectional_path),
        'BATS.morf_derrivational': process_benchmark_set(morf_derrivational_path),
        'BATS.sem_enciclopedic' :  process_benchmark_set(sem_enciclopedic_path),
        'BATS.sem_lexicographic' : process_benchmark_set(sem_lexicographic_path)
        }
    return analogy

def process_analogy_quads():
    data_path = os.environ["EMBEDDING_EVALUATION_DATA_PATH"]
    google_path = os.path.join(data_path, "google_dataset", "questions-words.txt")

    analogy = {
        'google' : process_benchmark_quads(google_path),
        }
    return analogy

def evaluate_pairs_3cosadd(a, a_prime, b, b_prime, E_mmodal, E_fallback, topk):
    """
    Evaluate using 3CosAdd
    """
    E = None
    if all(x in E_mmodal for x in (a, b, a_prime, b_prime)):
        E = E_mmodal
    elif all(x in E_fallback for x in (a, b, a_prime, b_prime)):
        E = E_fallback
    else:
        return None
    vec_a = E.get_vector(a)
    vec_a_prime = E.get_vector(a_prime)
    # compute predicted vector, and get NN words
    vec_b = E.get_vector(b)
    #vec_b_prime = E.get_vector(test_pair[1])
    vec_b_prime_predicted = vec_a_prime - vec_a + vec_b
    nn = E.get_nn_vector(vec_b_prime_predicted, topk, exclude_words = set((a, b, a_prime)))
    # see whether test_pair[1] is in vec_b_prime_predicted
    if b_prime in nn:
        return True
    return False

# evaluate using 3CosAvg

def evaluate_pairs_3cosavg(train_pairs, test_pair, E_mmodal, E_fallback, topk):
    """
    Evaluate using 3CosAvg

    Basically, compute the mean of all pairs in train, try to predict test.
    See ThreeCosAvg solver in vecto (https://github.com/vecto-ai/vecto.git)
    """
    E = None
    if all(x in E_mmodal for x in test_pair):
        E = E_mmodal
    elif all(x in E_fallback for x in test_pair):
        E = E_fallback
    else:
        return None
    # average all train pairs
    vecs_a = []
    vecs_a_prime = []
    for pair in train_pairs:
        if not all(x in E for x in pair):
            continue
        vecs_a.append(E.get_vector(pair[0]))
        vecs_a_prime.append(E.get_vector(pair[1]))
    if len(vecs_a) == 0:
        return None
    vec_a = np.vstack(vecs_a).mean(axis=0)
    vec_a_prime = np.vstack(vecs_a_prime).mean(axis=0)

    # compute predicted vector, and get NN words
    vec_b = E.get_vector(test_pair[0])
    #vec_b_prime = E.get_vector(test_pair[1])
    vec_b_prime_predicted = vec_a_prime - vec_a + vec_b
    nn = E.get_nn_vector(vec_b_prime_predicted, topk, exclude_words = set(test_pair[0]))
    # see whether test_pair[1] is in vec_b_prime_predicted
    if test_pair[1] in nn:
        return True
    return False

def evaluate_dataset_set(dataset, E, E_fallback, topk):
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# kf = KFold(n_splits=4)
# TRAIN: [1 2 3] TEST: [0]
# TRAIN: [0 2 3] TEST: [1]
# TRAIN: [0 1 3] TEST: [2]
# TRAIN: [0 1 2] TEST: [3]
    pairs = [ (k,v) for k,v in dataset.items() ]
    kfold = KFold(n_splits=len(pairs))
    cnt_splits = kfold.get_n_splits(pairs)
    loo = kfold.split(pairs)
    ok_n = 0
    sys_n = 0
    for train, test in loo:
        test_pair = pairs[test[0]]
        train_pairs = [pairs[i] for i in train]
        ok = evaluate_pairs_3cosavg(train_pairs, test_pair, E, E_fallback, topk)
        if ok is None:
            continue
        sys_n += 1
        if ok:
            ok_n += 1
    if sys_n == 0:
        prec = 0
    else:
        prec = ok_n / sys_n
    rec = ok_n / cnt_splits

    return {'prec' : prec, 'rec' : rec}


def evaluate_dataset_quads(dataset, E, E_fallback, topk):
    ok_n = 0
    sys_n = 0
    if len(dataset) == 0:
        return {'prec' : 0, 'rec' : 0}
    for (a, a_prime, b, b_prime) in dataset:
        ok = evaluate_pairs_3cosadd(a, a_prime, b, b_prime, E, E_fallback, topk)
        if ok is None:
            continue
        sys_n += 1
        if ok:
            ok_n += 1
    if sys_n == 0:
        prec = 0
    else:
        prec = ok_n / sys_n
    rec = ok_n / len(dataset)

    return {'prec' : prec, 'rec' : rec}

def words_in_benchmarks_set(all_benchmarks, bname=None):
    if bname == None:
        bname = set(all_benchmarks.keys())
    vocab = set()
    for benchmark_string in bname:
        benchmark = all_benchmarks[benchmark_string]
        for k,v in benchmark.items():
            vocab.add(k)
            vocab.add(v)
    return vocab

def words_in_benchmarks_quads(all_benchmarks, bname=None):
    if bname == None:
        bname = set(all_benchmarks.keys())
    vocab = set()
    for benchmark_string in bname:
        benchmark = all_benchmarks[benchmark_string]
        for (a, b, c, d) in benchmark:
            vocab.add(a)
            vocab.add(b)
            vocab.add(c)
            vocab.add(d)
    return vocab

class EvaluationAnalogy(object):

    def __init__(self):
        self.all_benchmarks = {
            'set' : process_analogy_set(),
            'quads' : process_analogy_quads()
        }

    def words_in_benchmarks(self, bname=None):
        return words_in_benchmarks_set(self.all_benchmarks['set']) | words_in_benchmarks_quads(self.all_benchmarks['quads'])

    def evaluate(self, my_embedding, embedding_fallback, topk = 1):
        if embedding_fallback == None:
            embedding_fallback = dict()
        results = {}
        E = EmbMatrix(my_embedding)
        E_fallback = EmbMatrix(embedding_fallback)
        res_set = {}
        for benchmark_string, benchmark in self.all_benchmarks['set'].items():
            res_set[benchmark_string] = evaluate_dataset_set(benchmark, E, E_fallback, topk)
        res_quads = {}
        for benchmark_string, benchmark in self.all_benchmarks['quads'].items():
            res_quads[benchmark_string] = evaluate_dataset_quads(benchmark, E, E_fallback, topk)
        return { 'set' : res_set, 'quads' : res_quads }
