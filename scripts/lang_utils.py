import pickle
import re

import bcolz
import numpy as np


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.n_words = 3  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs, max_src_len, max_tar_len = read_langs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, max_src_len, max_tar_len


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    src_lines = open(lang1, 'r').read().strip().split('\n')
    tar_lines = open(lang2, 'r').read().strip().split('\n')
    assert len(src_lines) == len(tar_lines)

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s), t] for s, t in zip(src_lines, tar_lines)]
    max_src_len = max([len(p[0]) for p in pairs])
    max_tar_len = max([len(p[1]) for p in pairs])

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs, max_src_len, max_tar_len


def normalize_string(s):
    """
    :param s: the string to be normalized.
    :return:
    """
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def vectors_for_input_lang(lang, glove_path):
    # source for this code: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    # glove_path = '/home/ng/workspace/lggltl/lggltl/glove.6B/'
    vectors = bcolz.open(glove_path + '6B.50.dat')[:]
    words = pickle.load(open(glove_path + '6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(glove_path + '6B.50_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    target_vocab = lang.index2word

    emb_dim = 50

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0
    i = 0

    for word in target_vocab:
        try:
            # print(target_vocab[word])
            weights_matrix[i] = glove[target_vocab[word]]
            # print(i)
            # print(word)
            words_found += 1
            i += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))

    return weights_matrix
