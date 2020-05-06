# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 10:47:47 2020

@author: jackc
"""
from Modules import adamGradientDescent as adam
import scipy.io
import numpy as np
import h5py
import matplotlib.pyplot as plt
from Modules import testCases_v4a
from Modules import dropout

def sigmoid_prime(x):
    s = 1/(1+np.exp(-x))
    #print(1-s)
    return s * (1-s)


def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A,cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def relu_prime(x):
    return np.where(x>0, 1.0, 0.0)


#e.g. [5,4,3]
def initialize_parameters(layer_dims):
    parameters = {}
    #layers
    L = len(layer_dims)
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layer_dims[l],1)) * np.sqrt(2./layer_dims[l-1])
    assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
    assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters

testParam = initialize_parameters([5,4,3])

def L_Model_Forward(X,parameters):
    caches = []
    A = X
    L = int(len(parameters)/2)
    for l in range(1,L):
        A,cache =linear_forward_Activation(A,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)
    AL,cache =linear_forward_Activation(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    return AL,caches

def L_Model_Forward_WithCost(X,Y,parameters,lambd):
    #print(parameters)
    caches = []
    A = X
    L = int(len(parameters)/2)
    for l in range(1,L):
        A,cache =linear_forward_Activation(A,parameters["W"+str(l)],parameters["b"+str(l)],"relu")
        caches.append(cache)
    AL,cache =linear_forward_Activation(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    J = compute_cost(AL,Y,parameters,lambd)
    return J,AL,caches
    
def linear_forward_Activation(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev,W,b)
    if activation == "sigmoid":
        A,activation_cache = sigmoid(Z)  
    else:
        A,activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A,cache

def linear_forward(A_prev,W,b):
    Z = np.dot(W,A_prev) + b
    linear_cache = {}
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    linear_cache = (A_prev,W,b)
    return Z, linear_cache

def compute_cost(AL, Y, parameters, lambd):
    m = Y.shape[1]
    #print("AL CHECK")
    #print(AL)
    #print(np.inf(AL.any()))
    # Compute loss from aL and y.
    cost = 1/m*-(np.sum(Y * np.log(AL) + (1-Y)*np.log(1-AL))) 

    regTerm = 0
    L = int(len(parameters) / 2)
    for l in range(L):
        #print(parameters["W"+str(l+1)])
        regTerm = regTerm + np.sum(np.square(parameters["W"+str(l+1)]))
        regTerm = regTerm * (lambd / 2) * (1/m) 
    cost = cost + regTerm
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    return cost

def linear_activation_backward(dA, cache, activation,lambd):
    linear_cache, activation_cache = cache
    Z = activation_cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    #m=211
  
    #grads["dZ"+str(L)] = grads["dA"+str(L)] * sigmoid_prime(Z)#should be AL - Y
    #grads["db"+str(L)] = (1/m) *  np.sum(grads["dZ"+str(L)],axis=1,keepdims=True)
    #grads["dW"+str(L)] = (1/m) *  grads["dZ"+str(L)].dot(A_prev.T)
    #grads["dA"+str(L-1)] = W.T.dot(grads["dZ"+str(L)])
    
    if activation == "relu":
        #relu_backward function will calculate  dA * relu'(Z)
        dZ = dA * relu_prime(Z) #activation_cache contains Z  
        #dA_prev, dW, db = linear_backward(dZ,linear_cache) #linear_cache contains A_prev, W, b
        
    elif activation == "sigmoid":
        #sigmoid(backward) function will calculate  dA * relu'(Z)
        dZ = dA * sigmoid_prime(Z) #activation_cache contains Z
        #dA_prev, dW, db = linear_backward(dZ,linear_cache) #linear_cache contains A_prev, W, b
        
    db = (1/m) *  np.sum(dZ,axis=1,keepdims=True)
    dW = (1/m) *  dZ.dot(A_prev.T) + (lambd / m) * W
    dA_prev = W.T.dot(dZ)

    
    return dA_prev, dW, db




def L_Model_Backwards(AL,Y,caches,lambd):
    grads = {}
    L = len(caches)
    #print(str(L))
    
    
    #last layer - SIGMOID
    
    cache = caches[L-1]
    #linear_cache, activation_cache =cache#A_prev,W,b | Z
    #A_prev, W, b = linear_cache
    #m = A_prev.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    #Z = activation_cache
    #grads["dA"+str(L)] = (AL-Y)/(AL*(1-AL))
    grads["dA"+str(L)] = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    #grads["dZ"+str(L)] = grads["dA"+str(L)] * sigmoid_prime(Z)#should be AL - Y
    #grads["db"+str(L)] = (1/m) *  np.sum(grads["dZ"+str(L)],axis=1,keepdims=True)
    #grads["dW"+str(L)] = (1/m) *  grads["dZ"+str(L)].dot(A_prev.T)
    grads["dA"+str(L-1)], grads["dW"+str(L)],  grads["db"+str(L)] = linear_activation_backward(grads["dA"+str(L)], cache,"sigmoid",lambd)
    # print(grads)

    #second last layer to first layer
    for l in reversed(range(L-1)):
        #RELU
        #print(l)
        cache = caches[l]
        #linear_cache, activation_cache =cache#A_prev,W,b | Z
        #A_prev, W, b = linear_cache
        #Z = activation_cache
        
        grads["dA"+str(l)], grads["dW"+str(l+1)],  grads["db"+str(l+1)] = linear_activation_backward(grads["dA"+str(l+1)], cache,"relu",lambd)
        #grads["dZ"+str(l+1)] = grads["dA"+str(l+1)] * relu_prime(Z)#should be AL - Y
        #grads["db"+str(l+1)] = (1/m) *  np.sum(grads["dZ"+str(l+1)],axis=1,keepdims=True)
        #grads["dW"+str(l+1)] = (1/m) *  grads["dZ"+str(l+1)].dot(A_prev.T)
        #grads["dA"+str(l)] = W.T.dot(grads["dZ"+str(l+1)])
        #print(grads)

    return grads



def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads['dW'+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads['db'+str(l+1)]
    return parameters


def L_layer_model_mini(X, Y, layers_dims, learning_rate = 0.3, num_iterations = 25000, print_cost=True, lambd = 0.7, mini_batch_size = 64, optimizer="gd", printIteration = 100, updateIteration = 50):#lr was 0.009
    
    alpha0 = learning_rate
    np.random.seed(1)
    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]  
    
    # Parameters initialization. (≈ 1 line of code)
    #e.g. layers_dims = [12288, 20, 7, 5, 1]
    parameters = initialize_parameters(layers_dims)
    
    if(optimizer=="adam"):
        v, s = adam.initialize_adam(parameters)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        seed = seed + 1

        minibatches = adam.random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:
            # Select a minibatch
        
            
            (minibatch_X, minibatch_Y) = minibatch
            #print(minibatch_X.shape)
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            #caches contains list of cache i.e (linearcache=(A_prev,W,B), activationcache= (Z))
    
            AL, caches = L_Model_Forward(minibatch_X, parameters)
            # Compute cost.
            #cost = compute_cost(AL,Y,parameters,lambd)
            # Compute cost and add to the cost total
            cost_total += adam.compute_costMini(AL, minibatch_Y)
            
        
            # Backward propagation.
            grads = L_Model_Backwards(AL,minibatch_Y,caches,lambd)
            #print(grads["dW2"])
            #print(grads["dA3"])
            # Update parameters.
            if(optimizer=="adam"):
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1 #should be 1e-8...
                t = t + 1 # Adam counter / mini-batch no
                parameters, v, s = adam.update_parameters_with_adam(parameters, grads, v, s,
                                                                   t, learning_rate, beta1, beta2,  epsilon)
            else:
                parameters = update_parameters(parameters,grads,learning_rate)
            
            #learning rate decay
            #learning_rate = (0.7 / np.sqrt(t)) * alpha0
            
        cost_avg = cost_total / m
        # Print the cost every 100 training example
        if print_cost and i % printIteration == 0:
            print ("Cost after iteration %i: %f" %(i, cost_avg))
            #print(learning_rate)
        if print_cost and i % updateIteration == 0:
            costs.append(cost_avg)

            
    # plot the cost
    #plt.cla()
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate = 0.3, num_iterations = 25000, print_cost=True, lambd = 0.7):#lr was 0.009
    "START"
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    #e.g. layers_dims = [12288, 20, 7, 5, 1]
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        #caches contains list of cache i.e (linearcache=(A_prev,W,B), activationcache= (Z))
        AL, caches = L_Model_Forward(X, parameters)
        # Compute cost.
        cost = compute_cost(AL,Y,parameters,lambd)
    
        # Backward propagation.
        grads = L_Model_Backwards(AL,Y,caches,lambd)

        # Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)

                
        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def L_layer_model_dropout(X, Y, layers_dims, learning_rate = 0.3, num_iterations = 25000, print_cost=True, lambd = 0.7):#lr was 0.009
    "START"
    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    #e.g. layers_dims = [12288, 20, 7, 5, 1]
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        #caches contains list of cache i.e (linearcache=(A_prev,W,B), activationcache= (Z))
        AL, caches = dropout.L_Model_Forward_Dropout(X, parameters,0.5)
        # Compute cost.
        cost = compute_cost(AL,Y,parameters,lambd)
    
        # Backward propagation.
        grads = dropout.L_Model_Backwards_Dropout(AL,Y,caches,lambd,0.5)

        # Update parameters.
        parameters = update_parameters(parameters,grads,learning_rate)

                
        # Print the cost every 100 training example
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 10 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



def load_2D_dataset():
    data = scipy.io.loadmat('data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    
    print("plotting")
    plt.scatter(x=train_X[0, :], y=train_X[1, :], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral);
    plt.show()
    
    return train_X, train_Y, test_X, test_Y

def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = L_Model_Forward(X, parameters)
    predictions = (a3>0.5)
    return predictions

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_Model_Forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p



