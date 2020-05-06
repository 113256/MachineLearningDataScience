# -*- coding: utf-8 -*-
from Modules import deepnn
import numpy as np


def initDropoutMatrix(A, keepprob):
    D = np.random.rand(A.shape[0],A.shape[1])  
    '''
    same as 
    
    for i,v in enumerate(x):
        if v < keep_prob:
            x[i] = 1
        else: # v >= keep_prob
            x[i] = 0
    '''
    D = (D < keepprob).astype(int)
    return D

def L_Model_Forward_Dropout(X,parameters, keepprob = 0.5):
    caches = []
    #dropoutVectors = {}
    A = X
    L = int(len(parameters)/2)
    for l in range(1,L):
        A,cache =deepnn.linear_forward_Activation(A,parameters["W"+str(l)],parameters["b"+str(l)],"relu")                                       # Step 1: initialize matrix D1 = np.random.rand(..., ...)
        D = initDropoutMatrix(A, keepprob)
        A = np.multiply(A,D)
        A = A / keepprob
        #dropoutVectors["D"+str(l)] = D #append D1,D2.. to the parameters cache
        bigCache = (cache,D)        
        #print(D.shape)
        caches.append(bigCache)
    #dont need dropout in final layer!!! or input layer
    AL,cache =deepnn.linear_forward_Activation(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    #print("END FP")
    caches.append(cache)
    return AL,caches



def L_Model_Backwards_Dropout(AL,Y,caches,lambd, keepprob):
    grads = {}
    L = len(caches)
    
    #no D vector in final layer as we dont need dropout in final layer
    cache = caches[L-1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    grads["dA"+str(L)] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA"+str(L-1)], grads["dW"+str(L)],  grads["db"+str(L)] = deepnn.linear_activation_backward(grads["dA"+str(L)], cache,"sigmoid",lambd)
    #print("2nd last layer - "+str(L-1))
    
    #second last layer to first layer
    for l in reversed(range(L-1)):
        cache,D = caches[l]
        #print("apply dropout 2nd last layer - "+str(l+1))
        # Apply mask D to shut down the same neurons as during the forward propagation
        grads["dA"+str(l+1)] = np.multiply(grads["dA"+str(l+1)],D)   
        grads["dA"+str(l+1)] = grads["dA"+str(l+1)] / keepprob         

 
        #linear_cache, activation_cache =cache#A_prev,W,b | Z
        #A_prev, W, b = linear_cache
        #Z = activation_cache
        
        grads["dA"+str(l)], grads["dW"+str(l+1)],  grads["db"+str(l+1)] = deepnn.linear_activation_backward(grads["dA"+str(l+1)], cache,"relu",lambd)
    return grads