#!/usr/bin/env python3
import os

def parse_csv(name, path):
    dataset = {}
    try:
        with open(path, "r") as f:
            data = f.read().splitlines()
        for line in data[1:]:
            s = line.split(",")
            word1 = s[1].lower()
            word2 = s[2].lower()
            score = float(s[3])
            dataset[(word1, word2)] = score
    except Exception as e:
        print("Error parsing {}: {}".format(name, e))
        exit(1)
    return dataset

def parse_vis_sem_sim(path):
    semsim = {}
    vissim = {}
    with open(path, "r") as f:
        data = f.read().splitlines()
    for line in data[1:-1]:
        s = line.split("\t")
        words = s[0].split("#")
        word1 = words[0].lower()
        word2 = words[1].lower()
        if "_(" in word1 or "_(" in word2:
            continue
        sems = float(s[1])
        viss = float(s[2])
        semsim[(word1, word2)] = sems
        vissim[(word1, word2)] = viss
    return vissim, semsim

def parse_men(men_path, lemma=True):
    men = {}
    with open(men_path, "r") as f:
        data = f.read().splitlines()

    for line in data:
        s = line.split(" ")
        if lemma:
            word1 = s[0][:-2].lower()
            word2 = s[1][:-2].lower()
        else:
            word1 = s[0].lower()
            word2 = s[1].lower()
        score = float(s[2])
        men[(word1, word2)] = score
    return men


def parse_simlex(simlex_path):
    simlex = {}
    usf = {}
    conq = [ {} for i in range(4) ] # 4 quartiles according to concreteness
    with open(simlex_path, "r") as f:
        data = f.read().splitlines()

    for line in data[1:]:
        s = line.split("\t")
        word1 = s[0].lower()
        word2 = s[1].lower()
        s999 = float(s[3])
        has_usf = s[8] == "1"
        usf_score = float(s[7])
        simlex[(word1, word2)] = s999
        if has_usf:
            usf[(word1, word2)] = usf_score
        conq[int(s[6]) - 1][(word1, word2)] = s999
    return usf, simlex, conq[0], conq[1], conq[2], conq[3]

def parse_wordsim353(wordsim353_path):
    ws353 = {}
    with open(wordsim353_path, "r") as f:
        data = f.read().splitlines()
    for line in data[1:]:
        s = line.split(",")
        word1 = s[0].lower()
        word2 = s[1].lower()
        score = float(s[2])
        ws353[(word1, word2)] = score
    return ws353

def parse_verb(verb_path):
    verb = {}
    with open(verb_path, "r") as f:
        data = f.read().splitlines()

    for line in data:
        s = line.split(" ")
        word1 = s[0].lower()
        word2 = s[1].lower()
        score = float(s[2])
        verb[(word1, word2)] = score
    return verb


def parse_rw(rw_path):
    rw = {}
    with open(rw_path, "r") as f:
        data = f.read().splitlines()
    for line in data:
        s = line.split("\t")
        word1 = s[0].lower()
        word2 = s[1].lower()
        score = float(s[2])
        rw[(word1, word2)] = score
    return rw

def parse_mturk(mturk_path):
    mturk = {}
    with open(mturk_path, "r") as f:
        data = f.read().splitlines()
    for line in data[1:]:
        s = line.split(",")
        word1 = s[0].lower()
        word2 = s[1].lower()
        score = float(s[2])
        mturk[(word1, word2)] = score
    return mturk



def process_benchmarks():
    data_path = os.environ["EMBEDDING_EVALUATION_DATA_PATH"]
    simlex_path = os.path.join(data_path, "SimLex-999/SimLex-999.txt")
    wordsim353_path = os.path.join(data_path, "wordsim/combined.csv")
    men_path_lemma = os.path.join(data_path, "men/MEN_dataset_lemma_form_full")
    men_path_natural = os.path.join(data_path, "men/MEN_dataset_natural_form_full")
    vis_sem_sim_path = os.path.join(data_path, "vis_sem_sim/similarity_judgements.txt")
    verb_path = os.path.join(data_path, "verb-143/en-verb-143.txt")
    rw_path = os.path.join(data_path, "rw/rw.txt")
    mturk771_path = os.path.join(data_path, "mturk-771/MTURK-771.csv")

    usf, simlex, simconq_q1, simconq_q2, simconq_q3, simconq_q4 = parse_simlex(simlex_path)
    ws353 = parse_wordsim353(wordsim353_path)
    men = parse_men(men_path_natural, lemma=False)
    vis_sim, sem_sim = parse_vis_sem_sim(vis_sem_sim_path)
    verb = parse_verb(verb_path)
    rw = parse_rw(rw_path)
    mturk771 = parse_mturk(mturk771_path)

    benchmarks = {"usf": usf,
	          "ws353": ws353,
	          "men":men,
	          "vis_sim":vis_sim,
	          "sem_sim":sem_sim,
	          "simlex":simlex,
                  "simlex-q1": simconq_q1,
                  "simlex-q2": simconq_q2,
                  "simlex-q3": simconq_q3,
                  "simlex-q4": simconq_q4,
                  "mturk771": mturk771,
                  "rw": rw
    }

    return benchmarks

if __name__ == "__main__":
    benchmarks = process_benchmarks()
    print(benchmarks["mturk_771"])

