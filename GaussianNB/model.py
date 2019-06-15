#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np


class GaussianNB:
    """
    Simple implementation of Gaussian Naive Bayes classifier
    
    Key Assumptions
    ----------------
    -Each feature is uncorrelated from each other
    -The values of the features are normally (gaussian) distributed. 
    
    Input
    ------
    var_smoothing (float): 
        Portion of the largest variance of all features 
        that is added to variances for calculation stability
    
    """
    def __init__(self, var_smoothing=1e-09):
        self.var_smoothing = var_smoothing

        
    def fit(self, X, y):
        """
        Fit Gaussian Naive Bayes according to X, y
        
        Input
        -----
        X: array, size: [n_samples, n_features] 
           Training vectors
            
        y: array-like, [n_samples,]
           Target values
        """
        
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        
        # If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance by epsilon, a small fraction of the standard
        # deviation of the largest dimension.
        self.epsilon = self.var_smoothing * np.var(X, axis=0).max()

        # Calculate the mean and variance of each feature for each class
        for i, c in enumerate(self.classes):
            # only select rows where the label equals the given class
            X_class = X[np.where(y==c)]
            self.parameters.append([])

            for col in X_class.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)
                
                    
    def _calculate_prior(self, c):
        """
        Calculate the prior of class c
        i.e. total number of samples in class c / 
             total number of samples
        
        Input
        -----
        c: class (integer)
        
        Output
        ------
        prior probability of class c (float)
        """
        return np.mean(self.y == c)
    
    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate the Gaussian likelihood of the data X given mean and var
        
        Output
        ------
        likelihood (float)
        """
        nom = np.exp(-(x - mean)**2/(2. * var + self.epsilon))
        denom = np.sqrt(2. * math.pi * var + self.epsilon)
        return nom / denom
    
    
    def _classify(self, sample):
        """
        Classifies the input sample according to the maximum posterior probability
        based on Bayes Rule P(Y/X) = P(X/Y)*P(Y)/P(X)
        
        P(Y/X) "Posterior" - the probability that the sample X is of class Y
        P(X/Y) "Likelihood"- the probability of data X given class distribution Y
        P(Y)   "Prior"     - the prior probability of class Y
        P(X)   "Marginal"  - scales the posterior to make it a proper probability distribution.
                             Since this value is often extremely difficult or impossible to find,
                             and it is not needed to find the maximum P(Y/X), it is ignored
                
        Input
        -----
        sample: input sample (numpy array)
        
        Output
        ------
        predicted class of sample (integer)
        """
        posteriors = []
        for i, c in enumerate(self.classes):
            
            posterior = self._calculate_prior(c)
            
            # posterior is a product of prior and likelihoods due to assumption of independent features
            for feature_value, params in zip(sample, self.parameters[i]):
                likelihood = self._calculate_likelihood(feature_value, params["mean"], params["var"])
                posterior *= likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]
    
    def predict(self, X):
        """
        Perform classificaiton on an array of test vectors X
        
        Input
        ------
        X: input samples (numpy array)
        
        Output
        ------
        Y: predicted classes (numpy array)
        """
        y_pred = [self._classify(sample) for sample in X]
        return y_pred
    
    def accuracy_score(self, y_true, y_pred):
        """
        Calculates accuracy of classifier
        
        Input
        -----
        y_true: input labels (numpy array)
        y_pred: predicted output (numpy array)


        Output
        ------
        accuracy: accuracy score
        """
        
        accuracy = np.sum(y_true == y_pred, axis=0)/len(y_true)
        return accuracy