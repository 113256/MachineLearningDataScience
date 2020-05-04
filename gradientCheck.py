# -*- coding: utf-8 -*-

from Modules import testCases_v4a
from Modules import gradientCheckHelper
from Modules import deepnn
import h5py

#grad check test

#USING DEEP NN TEST CASE
trainG_X, trainG_Y, testG_X, testG_Y = load_2D_dataset()
layers_dimsTest = [trainG_X.shape[0], 20, 3, 1]
parametersG = deepnn.initialize_parameters(layers_dimsTest)

'''
#USING SIMPLE TEST CASE
trainG_X, trainG_Y, parametersG = gradient_check_n_test_case()
trainG_Y = trainG_Y.reshape([1,3])
'''
J, AL, caches = deepnn.L_Model_Forward_WithCost(trainG_X, trainG_Y, parametersG,0)
gradsG = deepnn.L_Model_Backwards(AL, trainG_Y, caches,0)
difference, grad, gradapprox = gradientCheckHelper.gradient_check_n(parametersG, gradsG, trainG_X, trainG_Y)