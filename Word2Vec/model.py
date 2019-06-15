#!/usr/bin/env python
# coding: utf-8

import numpy as np
from collections import Counter

import string
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt


class Word2Vec:
    """
    Python implementation of Word2Vec from scratch.
    The Word2Vec class trains a fully connected neural network with one hidden layer,
    either using the skip-gram or the continuous bag of words model (cbow).
    """

    def __init__(self, method='cbow', window_size=2, n=10, epochs=1000, learning_rate=0.1):
        self.word_to_id = dict()
        self.id_to_word = dict()

        if method == "cbow":
            self.method = self._cbow
        elif method == 'skipgram':
            self.method = self._skipgram
        else:
            raise ValueError("Unrecognised Method. Provide one of the following:'skipgram','cbow'")

        self.window_size = window_size  # context window +/- center word
        self.n = n  # dimension of word embeddings
        self.epochs = epochs  # number of training iterations
        self.eta = learning_rate

        self.seed = np.random.seed(0)

    def _preprocess(self, corpus):
        """
        Transforms corpus to usable format by:
        1. Removing punctuation
        2. Tokenizing

        Input
        ------
        Corpus: array of strings e.g. ["hello there","how are you"] (numpy array)

        Returns
        -------
        Corpus_processed (numpy array)
        """
        corpus_processed = []

        for sentence in corpus:
            sentence = sentence.replace("-", " ").lower()
            sentence = sentence.translate(str.maketrans('', '', string.punctuation))

            corpus_processed.append(word_tokenize(sentence))
        return np.array(corpus_processed)

    def _mapping(self, corpus_processed):
        """
        Generates word to word id and word id to word mappings (dictionaries)

        Input
        ------
        Corpus_processed: processed corpus of text (numpy array)
        """
        word_counts = Counter()

        for sentence in corpus_processed:
            word_counts.update(sentence)

        for i, word in enumerate(word_counts.keys()):
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        self.v = len(self.word_to_id)
        return

    def _onehotvec(self, word):
        """
        Generates one-hot vector of word in vocabulary

        Input
        ------
        Word: word from vocabulary e.g. "hello" (string)

        Output
        -------
        Word_vec: one-hot vector representation of word (list)
        """
        if word in self.word_to_id.keys():
            word_vec = [0 for i in range(0, self.v)]
            word_index = self.word_to_id[word]
            word_vec[word_index] = 1
            return word_vec
        else:
            raise ValueError("Word does not exist in the vocabulary")

    def generate_training_data(self, corpus):
        """
        Generates training data, in the form of a binary matrix
        with one-hot vector representations of each word pair
        within the window size

        Input
        ------
        Corpus: array of strings e.g. ["hello there","how are you"] (numpy array)

        Output
        -------
        Training_data (numpy array)
        """
        if len(corpus) == 0:
            raise ValueError('You need to specify a corpus of text.')

        training_data = []

        corpus_processed = self._preprocess(corpus)

        self._mapping(corpus_processed)

        for sentence in corpus_processed:
            for i, word in enumerate(sentence):
                w_t = self._onehotvec(sentence[i])

                lower_bound = max(0, i - self.window_size)
                upper_bound = min(len(sentence), i + self.window_size + 1)

                window_index = list(range(lower_bound, i)) + \
                               list(range(i + 1, upper_bound))
                w_c = []

                for j in window_index:
                    w_c.append(self._onehotvec(sentence[j]))
                training_data.append([w_t, w_c])

        return np.array(training_data)

    def _softmax(self, x):
        """
        Normalizes vector into a probability distribution. Each
        element of the vector will be compressed to a range [0,1]
        while the sum of all elements will be equal to 1.

        Input
        ------
        x (numpy array)

        Output
        -------
        Softmax of input vector (numpy array)
        """
        e_x = np.exp(x - np.max(x))  # max(x) is substracted for numerical stability
        return e_x / e_x.sum(axis=0)

    def _forwardprop(self, x):
        """
        Forward propagation of input in Neural Network

        Input
        ------
        x (numpy array)

        Output
        -------
        y: output vector giving the probability that each word in the
           vocabulary appears near the input word (numpy array)
        h: hidden layer, equivalent to the vector representation of
           the input word (numpy array)
        u: vector measuring the similarity between the context and the
           target wordd (numpy array)
        """
        h = np.dot(self.w1.T, x)  # shape  n x 1
        u = np.dot(self.w2.T, h)  # shape v x 1
        yhat = self._softmax(u)  # shape v x 1
        return yhat, h, u

    def _cbow(self, w_t, w_c, loss):
        """
        One training iteration of CBOW model.

        The CBOW model predicts a word (w_t) given a context (w_c).
        Context is fed at the input and the output layer of the
        neural network is a multinomial distribution describing
        the probability of the target word being the center word.

        Input
        ------
        w_t: one-hot vector representation of target word (list)
        w_c: one-hot vector representation(s) of context word(s) in the window (list)
        loss: carries the value of the training loss from the previous iteration (float)

        Output
        ------
        loss: updated value of training loss (float)
        """
        x = np.mean(w_c, axis=0)

        # FORWARD PROPAGATION
        yhat, h, u = self._forwardprop(x)

        # PREDICTION ERROR
        e = yhat - w_t

        # BACK PROPAGATION
        dw2 = np.outer(h, e)
        dw1 = np.outer(x, np.dot(self.w2, e))

        self.w1 += - self.eta * dw1
        self.w2 += - self.eta * dw2

        # TRAINING LOSS
        loss += -u[w_t.index(1)] + np.log(np.sum(np.exp(u)))
        return loss

    def _skipgram(self, w_t, w_c, loss):
        """
        One training iteration of Skip-Gram model.

        The Skip-Gram model predicts the context given an input word.
        A target word is fed at the input and the output layer of the
        neural network is replicated multiple times to match the chosen
        number of context words (window_size). The output is therefore
        multiple multinomial distributions.

        Input
        ------
        w_t: one-hot vector representation of target/input word (list)
        w_c: one-hot vector representation(s) of context word(s) in the window (list)
        loss: carries the value of the training loss from the previous iteration (float)

        Output
        ------
        loss: updated value of training loss (float)

        """
        # FORWARD PROPAGATION
        yhat, h, u = self._forwardprop(w_t)

        # PREDICTION ERROR
        e = np.array([yhat - word for word in w_c])
        EI = np.sum(e, axis=0)  # shape v x 1

        # BACK PROPAGATION
        dw2 = np.outer(h, EI)
        dw1 = np.outer(w_t, np.dot(self.w2, EI.T))

        self.w1 += - self.eta * dw1
        self.w2 += - self.eta * dw2

        # TRAINING LOSS
        loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
        return loss

    def train(self, training_data):
        """
        Main method of the word2vec class where training takes place

        Input
        ------
        training_data: binary array with one-hot vector representations
                       of each word pair (numpy array)

        Output
        -------
        Plot of training loss against training iteration
        """
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v, self.n))
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v))

        losses = []

        for i in range(self.epochs):
            loss = 0

            for w_t, w_c in training_data:
                loss = self.method(w_t, w_c, loss)

            losses.append(loss)

            if i % 10 ** (len(str(self.epochs)) - 2) == 0:
                print("Cost after epoch {}: {}".format(i, loss))

        print("Cost after epoch {}: {}".format(i, loss))
        plt.plot(np.arange(self.epochs), losses)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')
        plt.title(self.method.__name__.replace("_", ""))
        plt.show()
        return

    def word_vec(self, word=None, returnall=False):
        """
        Returns trained word embeddings

        Input
        ------
        words (list of strings)
        returnall: if True, return word embedding matrix
                   if False and a word/words are specified,
                   return word embedding of word
        Output
        word_vec: word embedding
        w1: word embedding matrix
        """
        if returnall:
            return self.w1
        elif not returnall:
            if word in self.word_to_id.keys():
                word_index = self.word_to_id[word]
                word_vec = self.w1[word_index]
                return word_vec
            else:
                raise ValueError("Word does not exist in the vocabulary")

    def nearest_words(self, word, number_of_words=2):
        """
        Predicts nearest words

        Input
        ------
        word: target/center word (string)
        number_of_words: number of adjacent word predictions

        Output
        -------
        top_words_scores: nearest words with probability scores
        """
        top_words_scores = []

        if word.lower() in self.word_to_id.keys():
            if number_of_words >= len(self.word_to_id):
                print(
                    "warning - the requested number of predictions exceeds the number of unique words in the vocabulary")

            word_vec = self._onehotvec(word.lower())
            y, h, u = self._forwardprop(word_vec)
            word_ranks = np.argsort(-y)[:number_of_words]

            for rank in word_ranks:
                top_words_scores.append([self.id_to_word[rank], round(y[rank], 3)])
            return top_words_scores
        else:
            raise ValueError("Word does not exist in the vocabulary")

    def similar_words(self, word, number_of_words=2):
        """
        Returns most similar words as defined by the
        cosine similarity of the trained word embeddings

        Input
        ------
        word (string)
        number_of_words: number of similar words to return

        Output
        -------
        top_words_scores: nearest words with probability scores
        """
        top_words_scores = []

        word_vec = self.word_vec(word)

        word_vec_norm = np.linalg.norm(word_vec)
        w1_norm = np.linalg.norm(self.w1, axis=1)

        cos_sim = np.dot(self.w1, word_vec) / (w1_norm * word_vec_norm)
        word_ranks = np.argsort(-cos_sim)[:number_of_words]

        for rank in word_ranks:
            top_words_scores.append([self.id_to_word[rank], round(cos_sim[rank], 3)])

        return top_words_scores