#!/usr/bin/env python3

import os
import argparse
from embedding_evaluation.evaluate_analogy import EvaluationAnalogy

import json as jsonmod
import numpy as np

def emb_name(emb_path):
    bname = os.path.basename(emb_path)
    return os.path.splitext(bname)[0]

def load_embeddings(path):
    try:
        with open(path) as fh:
            E = jsonmod.load(fh)
    except IOError as e:
        print("Can not load embeddings: {}".format(e))
        exit(1)
    embeddings = {}
    for word, v in E.items():
        embeddings[word.lower()] = np.array(v)
    return embeddings

def load_glove_embeddings(emb_path, vocab):
    embeddings = {}
    try:
        for line in open(emb_path, 'r', encoding='utf-8', errors='surrogateescape'):
            auxV = line.strip().split(' ')
            if auxV[0] not in vocab:
                continue
            embeddings[auxV[0].lower()] = np.array([ float(x) for x in auxV[1:] ])
    except IOError as e:
        print(e)
        exit(1)
    return embeddings

def main():
    os.environ["EMBEDDING_EVALUATION_DATA_PATH"] = "/tartalo03/users/muster/embedding_evaluation_paper/EloiZ/data/analogy"
    parser = argparse.ArgumentParser()
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('wemb',
                        help = "Word embeddings", nargs='+')
    parser.add_argument('--nofallback', action='store_true',
                        help = "Specify whether glove is used as fallback")
    parser.add_argument('--topk', type=int, default=10,
                        help = "OK if word is among nearest topK")
    opt = parser.parse_args()
    print(opt)

    all_results = {}
    eval_object = EvaluationAnalogy() # use EloiZ to evaluate embeddings

    E_fallback = None
    if not opt.nofallback:
        for emb_path in opt.wemb:
            bname = os.path.basename(emb_path)
            emb_name = os.path.splitext(bname)[0]
            if emb_name != 'glove':
                continue
            E_fallback = load_glove_embeddings(emb_path, eval_object.words_in_benchmarks())
        if E_fallback is None:
            import sys
            print('No glove embeddings (use --nofallback for no glove)', file=sys.stderr)
            exit(0)

    for emb_path in opt.wemb:
        bname = os.path.basename(emb_path)
        emb_name = os.path.splitext(bname)[0]
        if emb_name == 'glove':
            if E_fallback is not None:
                E = E_fallback
            else:
                E = load_glove_embeddings(emb_path, eval_object.words_in_benchmarks())
        else:
            E = load_embeddings(emb_path)
        all_results[emb_name] = eval_object.evaluate(E, E_fallback, opt.topk)
    dataset_names = ['']
    for emb_name, total_results in all_results.items():
        for ana_type in sorted(total_results.keys()):
            results = total_results[ana_type]
            dataset_names += sorted(total_results[ana_type].keys())
            dataset_names.append('-')
        break
    columns = [ dataset_names ]
    for emb_name, total_results in all_results.items():
        c = [ emb_name ]
        for ana_type in sorted(total_results.keys()):
            results = total_results[ana_type]
            for d in sorted(results.keys()):
                r = results.get(d, None)
                c.append("{:.2f} : {:.2f}".format(r['prec'] * 100, r['rec'] * 100))
            c.append('|')
        columns.append(c)
    rows = list(map(list, zip(*columns))) # transpose
    print("|" + "|".join(rows[0]) + "|")
    print("|-")
    for r in rows[1:]:
        print("|" + "|".join(r) + "|")

if __name__ == '__main__':
    main()
