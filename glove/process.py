"""This is from https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76"""

import os.path as osp
import pickle

import bcolz
import numpy as np

words = []
idx = 0
word2idx = {}
glove_data_path = osp.join(osp.dirname(__file__))
vectors = bcolz.carray(np.zeros(1), rootdir=osp.join(glove_data_path, '6B.50.dat'), mode='w')

with open(osp.join(glove_data_path, 'glove.6B.50d.txt'), 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float32)
        vectors.append(vect)

vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=osp.join(glove_data_path, '6B.50.dat'), mode='w')
vectors.flush()
pickle.dump(words, open(osp.join(glove_data_path, '6B.50_words.pkl'), 'wb'))
pickle.dump(word2idx, open(osp.join(glove_data_path, '6B.50_idx.pkl'), 'wb'))
