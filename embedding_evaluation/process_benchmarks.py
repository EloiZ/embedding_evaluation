import os

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
    return usf, simlex

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

def process_benchmarks():
    data_path = os.environ["EMBEDDING_EVALUATION_DATA_PATH"]
    simlex_path = os.path.join(data_path, "SimLex-999/SimLex-999.txt")
    wordsim353_path = os.path.join(data_path, "wordsim/combined.csv")
    men_path_lemma = os.path.join(data_path, "men/MEN_dataset_lemma_form_full")
    men_path_natural = os.path.join(data_path, "men/MEN_dataset_natural_form_full")
    vis_sem_sim_path = os.path.join(data_path, "vis_sem_sim/similarity_judgements.txt")

    usf, simlex = parse_simlex(simlex_path)
    ws353 = parse_wordsim353(wordsim353_path)
    men = parse_men(men_path_natural, lemma=False)
    vis_sim, sem_sim = parse_vis_sem_sim(vis_sem_sim_path)
    benchmarks = {"usf": usf,
	            "ws353": ws353,
	            "men":men,
	            "vis_sim":vis_sim,
	            "sem_sim":sem_sim,
	            "simlex":simlex}

    return benchmarks

