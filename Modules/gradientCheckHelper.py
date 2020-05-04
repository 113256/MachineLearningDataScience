# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import h5py
import matplotlib.pyplot as plt
from Modules import deepnn

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    shapes = {}
    for key,value in parameters.items():
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        shapes[key] = value.shape 
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1
    #keys tell us which vectors (W1,W2,b1,etc...) each index of theta belongs to
    #e.g. if indexes 0,1,2 of keys = "W1", mean W1 comprises theta[0], theta[1], theta[2]
    #shapes is a dictionary of the shapes of each vector W1,b1,W2
    return theta, np.array(keys).reshape(len(keys),1), shapes

def vector_to_dictionary(thetaAndKeys, shapes):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    '''
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))
    '''
    #concat = np.concatenate()
    for key,shapeValue in shapes.items():
        '''
        thetaAndKeys consists of:
        0.1 W1
        0.2 W1
        0.1 b1 
        etc....
        We want to filter the by W1, W2, b1..etc so that the array only contains values for only W1, only b1, and so on
        '''
        boolArray = thetaAndKeys[:,1] == key

        #print(boolArray)
        filteredArray = thetaAndKeys[boolArray]#filter away rows whose 2nd column dont contain key
        filteredArray = filteredArray[:,0]#remove 2nd column
        filteredArray = np.reshape(filteredArray,shapeValue)#reshape the 1-d array to original shape
        #print(filteredArray.shape)
        parameters[key] = filteredArray.astype(np.float)
    
    return parameters

def gradients_to_vector(gradients,shapes):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    count = 0
    #for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
    #MAKE SURE YOU LOOP IN THE CORRECT ORDER - I.E. SAME ORDER AS PARAMETERS i.e. W1,b1,W2,b2...
    #edit - just pass in shape array and use it...(shape is a dictionary)

    orderedGradients = []
    for k in shapes:
        orderedGradients.append("d"+k)
    print(orderedGradients)
    for key in orderedGradients:
        if "dA" in key:
            #print("c")
            #skip dA1, dA2 etc... because we dont want dA, we only want dW, db in this single vector
            #we will compare this single vector with gradapprox, which contains approximations of only dW and db too
            continue
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta
    
#TEST HELPER FUNCTIONS
'''
parameters = initialize_parameters([5,4,3])
theta,keys,shapes = dictionary_to_vector(parameters)
thetaAndKeys = np.hstack((theta,keys))
newParam = vector_to_dictionary(thetaAndkeys,shapes)
testGrad = {"dZ3": 1, "dW3": 4, "db3": 7,
                 "dA2": 2, "dZ2": 5, "dW2": 8, "db2": 0,
                 "dA1": 3, "dZ1": 6, "dW1": 9, "db1": 0}
gradients_to_vector(testGrad)
'''


def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    #parameters contains W1,b1,W2,b2,W3,b3
    #converts the "parameters" dictionary into a vector called "values", obtained by reshaping all parameters (W1, b1, W2, b2, W3, b3) into vectors and concatenating them
    #shape of parameters_values = (47,1) (concat and squeeze values of W1,b1,W2,b2...)
    parameters_values, keys, shapes = dictionary_to_vector(parameters)
    print(parameters_values.shape)
    #converted the "gradients" dictionary into a vector "grad" using gradients_to_vector().
    grad = gradients_to_vector(gradients,shapes)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    #loop through every element of W1,b1,W2,b2..
    for i in range(num_parameters):
        
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)                                      # Step 1
        #thetaplus[i][0] is one of the elements of parameter_values 
        thetaplus[i][0] = thetaplus[i][0] + epsilon                                # Step 2
        #turns thetaplus back to dictionary with keys "W1", "b1", "W2",....
        J_plus[i], _ , _= deepnn.L_Model_Forward_WithCost(X, Y, vector_to_dictionary(np.hstack((thetaplus,keys)),shapes),0)                                  # Step 3
        ### END CODE HERE ###
        
        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)                                        # Step 1
        thetaminus[i][0] = thetaminus[i][0] - epsilon                                      # Step 2        
        J_minus[i], _ , _ = deepnn.L_Model_Forward_WithCost(X, Y, vector_to_dictionary(np.hstack((thetaminus,keys)),shapes),0)                                   # Step 3
        ### END CODE HERE ###
        
        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
        ### END CODE HERE ###
    
    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)                                     # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                   # Step 2'
    difference = numerator / denominator                                              # Step 3'
    ### END CODE HERE ###

    print("Difference= "+str(difference))
    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference, grad, gradapprox



def gradient_check_n_test_case(): 
    np.random.seed(1)
    x = np.random.randn(4,3)
    y = np.array([1, 1, 0])
    W1 = np.random.randn(5,4) 
    b1 = np.random.randn(5,1) 
    W2 = np.random.randn(3,5) 
    b2 = np.random.randn(3,1) 
    W3 = np.random.randn(1,3) 
    b3 = np.random.randn(1,1) 
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    
    return x, y, parameters