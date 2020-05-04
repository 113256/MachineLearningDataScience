# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from Modules import deepnn as deepnn
import numpy as np
#from deepnn import *

def predictMinst(X, Y, parameters):
    #print(X.shape)
    # Forward propagation
    AL, caches = deepnn.L_Model_Forward(X, parameters)
    print(AL.shape)
    pred = np.argmax(AL, axis=0)
    correct = [1 if (a == b) else 0 for (a, b) in zip(pred, Y)]
    accuracy = sum(correct) / len(correct)
    
    print("Accuracy: "  + str(accuracy))
    return pred
    
    
def displayPrediction(features, parameters):
    #print one example
    demo = np.atleast_2d(features).reshape(28,28)
    plt.imshow(demo, cmap="hot", vmin=0, vmax=1) #reshape back to 20 pixel by 20 pixel
    plt.show()
    pred = deepnn.L_Model_Forward(features, parameters)
    pred = np.argmax(pred[0], axis=0)
    print(pred)
    return pred