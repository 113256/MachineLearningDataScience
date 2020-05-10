# -*- coding: utf-8 -*-

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from Modules import mnistHelper as mHelper
from Modules import deepnn as deepnn
from Modules import adamGradientDescent as adam
from Modules import testCases_v4a
import numpy as np
import matplotlib.pyplot as plt

import h5py

def load_dataset():
    train_dataset = h5py.File('Datasets/Handsigns/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('Datasets/Handsigns/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

trainXOriginal, trainyOriginal, testXOriginal, testyOriginal, _ = load_dataset()

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainXOriginal.shape, trainyOriginal.shape))
print('Test: X=%s, y=%s' % (testXOriginal.shape, testyOriginal.shape))

#print one example
demo = trainXOriginal[0]
print(trainXOriginal[0].shape)
plt.imshow(demo, cmap="hot", vmin=0, vmax=1) #reshape back to 20 pixel by 20 pixel
plt.show()

#one-hot encode
ytmp= np.reshape(trainyOriginal, (-1, 1))
encoder = OneHotEncoder(sparse=False)
yTrain = encoder.fit_transform(ytmp)
ytmp= np.reshape(testyOriginal, (-1, 1))
encoder = OneHotEncoder(sparse=False)
yTest = encoder.fit_transform(ytmp)

X_train_flatten = trainXOriginal.reshape(trainXOriginal.shape[0], -1)
X_test_flatten = testXOriginal.reshape(testXOriginal.shape[0], -1)

XTrain = X_train_flatten.astype('float32')
XTest = X_test_flatten.astype('float32')
#XTrain = X_train_flatten
#XTest = X_test_flatten
XTrain /= 255
XTest /= 255

#transpose all
XTest,XTrain,yTest,yTrain = XTest.T,XTrain.T,yTest.T,yTrain.T

print(yTrain.shape)
print(XTrain.shape)
#parameters = deepnn.L_layer_model(XTrain, yTrain,layers_dims = [XTrain.shape[0], 128, 6], num_iterations = 5000, learning_rate = 0.1, lambd = 0.3)
parameters = deepnn.L_layer_model_mini(XTrain, yTrain,layers_dims = [XTrain.shape[0], 25, 6], num_iterations = 1000, learning_rate =0.0001, lambd = 0, mini_batch_size=32, optimizer="gd", printIteration=100, updateIteration=5)
#parameters = deepnn.L_layer_model_mini(XTrain, yTrain,layers_dims = [XTrain.shape[0], 50, 10], num_iterations = 20, learning_rate = 0.8, lambd = 0, mini_batch_size=64, optimizer="adam",printIteration=2 ,updateIteration=2)
#parameters = deepnn.L_layer_model_dropout(XTrain, yTrain,layers_dims = [XTrain.shape[0], 64, 10], num_iterations = 70, learning_rate = 0.8)

def predictAcc(X, Y, parameters):
    #print(X.shape)
    # Forward propagation
    AL, caches = deepnn.L_Model_Forward(X, parameters)
    print(AL.shape)
    pred = np.argmax(AL, axis=0)
    print(pred.shape)
    yTmp = Y.reshape([1080,1]).ravel()
    correct = [1 if (a == b) else 0 for (a, b) in zip(pred, yTmp)]
    accuracy = sum(correct) / len(correct)
    
    print("Accuracy: "  + str(accuracy))
    return pred, yTmp








print ("On the training set:")
predictions_train, yt = predictAcc(XTrain, trainyOriginal, parameters)
print ("On the test set:")
#predictions_test = predictAcc(XTest, testyOriginal, parameters)
#pred3 = mHelper.displayPrediction(np.atleast_2d(XTrain[:,200]).T,parameters)

