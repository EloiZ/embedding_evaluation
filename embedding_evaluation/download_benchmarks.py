#!/usr/bin/env python3
import os

def download_hyperlex():
    hyperlex_url = 'http://people.ds.cam.ac.uk/iv250/paper/hyperlex/hyperlex-data.zip'
    os.system("wget -O data/hyperlex.zip %s" % hyperlex_url)
    os.system("unzip data/hyperlex.zip -d data/hyperlex")
    os.system("rm data/hyperlex.zip")

def download_simlex():
    simlex_url = "https://www.cl.cam.ac.uk/~fh295/SimLex-999.zip"
    os.system("wget -O data/simlex.zip %s" % simlex_url)
    os.system("unzip data/simlex.zip -d data/")
    os.system("rm data/simlex.zip")

def dowload_wordsim():
    wordsim_url = "http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip"
    os.system("mkdir data/wordsim")
    os.system("wget -O data/wordsim/wordsim.zip %s" % wordsim_url)
    os.system("unzip data/wordsim/wordsim.zip -d data/wordsim/")
    os.system("rm data/wordsim/wordsim.zip")

def download_men():
    men_lemma_url = "https://raw.githubusercontent.com/julieweeds/MEN/master/data/MEN/MEN_dataset_lemma_form_full"
    men_natural_url = "https://raw.githubusercontent.com/julieweeds/MEN/master/data/MEN/MEN_dataset_natural_form_full"
    os.system("mkdir data/men")
    os.system("wget -P data/men/ %s" % men_lemma_url)
    os.system("wget -P data/men/ %s" % men_natural_url)

#def download_vis_sim_sim():
#    vis_sem_sim_url = "http://homepages.inf.ed.ac.uk/s1151656/xyzblabli/similarity_judgements.txt"
#    os.system("mkdir data/vis_sem_sim")
#    os.system("wget -P data/vis_sem_sim/ %s" % vis_sem_sim_url)

def download_mcrae():
    mc_rae_url = "https://sites.google.com/site/kenmcraelab/norms-data/CONCS_FEATS_concstats_brm.xlsx"
    os.system("mkdir data/mc_rae")
    os.system("wget -P data/mc_rae/ %s" % mc_rae_url)

if __name__ == "__main__":
    os.system("mkdir data")
    download_simlex()
    dowload_wordsim()
    download_men()
    #download_vis_sim_sim()
    #download_mcrae()
