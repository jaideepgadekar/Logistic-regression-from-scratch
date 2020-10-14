#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:14:34 2020

@author: jaideep
"""


import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures


def import_data():
    X = np.genfromtxt('train_X_lg_v2.csv', dtype = 'float64',\
                      skip_header = 1 ,delimiter= ',')
    Y = np.genfromtxt('train_Y_lg_v2.csv', dtype = np.int32,\
                      delimiter = ',')
    Y = Y.reshape(len(X), 1)
    return X,Y

def initialize_weights(X):
    W = np.zeros((X.shape[1], 1))
    b = 0
    return W, b

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1.0 / (1.0 + np.exp(-x))

def cost_function( X,Y, W,b):
    m = len(X)
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    cost = (-1.0/m)*(np.sum(Y*np.log(A) + (1 - Y)*np.log(1 - A)))
    return cost

def compute_gradient_of_cost_function(X,Y,W,b):
    m = len(X)
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    dz = A - Y 
    dW = (1/m)*np.dot(X, dz.T)
    db = (1/m)*np.sum(dz)
    return dW, db
    

def optimise_weights(X,Y,W,b, num_iter,learning_rate):
    for i in range(num_iter):
        dW, db = compute_gradient_of_cost_function(X, Y, W,b)
        W -= learning_rate*dW
        b -= learning_rate*db
        #cost = cost_function(X,Y,W,b)
        #print(i, cost)
    return W, b

def data_per_class(X,Y, class_label):
    class_X = np.copy(X)
    class_Y = np.copy(Y)
    class_Y = np.where(class_Y == class_label, 1, 0)
    return class_X, class_Y

def accuracy(Y, pred_Y):
    acc = 0
    for  i in range(len(Y)):
        if(Y[i] == pred_Y[i]):
            acc += 1
    return acc/len(Y)            

def predict_values(X,W, b):
    pred_Y  = np.dot(W.T, X) + b
    pred_Y = pred_Y.T
    pred_Y = np.where(pred_Y >= 0.5, 1, 0)
    return pred_Y
if __name__ == '__main__':
    train_X,train_Y = import_data()
    poly = PolynomialFeatures(2)
    #train_X = poly.fit_transform(train_X)
    class_X, class_Y = data_per_class(train_X, train_Y, 0)
    W0, b0 = initialize_weights(class_X)
    #train_X, test_X, train_Y, test_Y = train_test_split(class_X,class_Y, test_size = 0.2)
    W0, b0 = optimise_weights(class_X.T, class_Y.T, W0, b0, 130000, 1e-4)
    
    print(accuracy(class_Y, predict_values(class_X.T,W0, b0)))

    class_X, class_Y = data_per_class(train_X, train_Y, 1)
    #class_X = poly.fit_transform(class_X)
    W1, b1 = initialize_weights(class_X)
    #train_X, test_X, train_Y, test_Y = train_test_split(class_X,class_Y, test_size = 0.2)
    W1, b1 = optimise_weights(class_X.T, class_Y.T, W1, b1, 40000, 0.0001)
    print(accuracy(class_Y, predict_values(class_X.T,W1, b1)))
    
    class_X, class_Y = data_per_class(train_X, train_Y, 2)
    W2, b2 = initialize_weights(class_X)
    #train_X, test_X, train_Y, test_Y = train_test_split(class_X,class_Y, test_size = 0.2)
    W2, b2 = optimise_weights(class_X.T, class_Y.T, W2, b2, 400, 1e-5)
    print(accuracy(class_Y, predict_values(class_X.T,W2, b2)))
    
    class_X, class_Y = data_per_class(train_X, train_Y, 3)
    W3, b3 = initialize_weights(class_X)
    #train_X, test_X, train_Y, test_Y = train_test_split(class_X,class_Y, test_size = 0.2)
    W3, b3 = optimise_weights(class_X.T, class_Y.T, W3, b3, 130000, 1e-4)
    print(accuracy(class_Y, predict_values(class_X.T,W3, b3)))
        

    weights = [[W0, b0], [W1,b1], [W2,b2], [W3,b3]]
    weights = np.array(weights)
    with open('weights.pkl', 'wb') as f:
        pickle.dump(weights, f)
    
  
    
    

    
    