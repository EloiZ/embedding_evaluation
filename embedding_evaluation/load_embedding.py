import pickle
import numpy as np


def load_embedding_pickle(pickle_path):
    with open(pickle_path, "rb") as f:
        embedding_dict = pickle.load(f)
    return embedding_dict

def load_embedding_npy(npy_path, vocab_path):
    npy_tensor = np.load(npy_path)
    return load_embedding_matrix(npy_tensor, vocab_path)

def load_embedding_matrix(npy_tensor, vocab_path):
    with open(vocab_path, "r") as f:
        vocab = f.read().splitlines()
    #assert len(vocab) == npy_tensor.shape[0]
    embedding_dict = {word: npy_tensor[i, :] for i, word in enumerate(vocab)}
    return embedding_dict

def load_embedding_textfile(textfile_path, sep=","):
    with open(textfile_path, "r") as f:
        lines = f.read().splitlines()
    embedding_dict = {}
    for line in lines:
        line = line.split(sep)
        embedding_dict[line[0]] = np.array(list(map(float, line[1:])))
    return embedding_dict

