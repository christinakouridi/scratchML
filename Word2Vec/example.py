#!/usr/bin/env python
# coding: utf-8

import numpy as np

from model import Word2Vec


def main():

    # ----DATA PREPARATION---- #
    corpus = np.array(['the quick brown fox jumped over the lazy dog'])

    # ----MODEL TRAINING---- #
    hyperparameters = {
        'method': "cbow",
        'window_size': 2,
        'n': 100,  # typically ranges from 100 to 300
        'epochs': 10000,
        'learning_rate': 0.01
    }

    cbow = Word2Vec(**hyperparameters)

    training_data = cbow.generate_training_data(corpus)

    cbow.train(training_data)

    # get word embedding of word
    print(cbow.word_vec("fox"))

    # get similar words
    print(cbow.similar_words("fox"))


if __name__ == "__main__":
    main()
