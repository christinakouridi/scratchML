#!/usr/bin/env python
# coding: utf-8

import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from model import GaussianNB


def normalize(X, norm = "l2"):
    """ Normalize the dataset X """
    if norm == "l1":
        n = np.atleast_1d(np.linalg.norm(X, 1, -1))
    elif norm == "l2":
        n = np.atleast_1d(np.linalg.norm(X, 2, -1))
    elif norm == "max":
        n = np.max(X, axis = -1)
    else:
        raise ValueError("invalid normalisation method provided")
    
    n[n == 0] = 1
    return X / np.expand_dims(n, -1)


def main():
    data = datasets.load_iris()
    
    X = data.data
    y = data.target
    
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = clf.accuracy_score(y_test, y_pred)
    
    print("model accuracy:", accuracy)
    return y_pred
    

if __name__ == "__main__":
    main()