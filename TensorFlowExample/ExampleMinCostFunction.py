# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

#this is just for print purpose - print out the output of cost function passing in updated w
def calculateCost(w): 
	return w ** 2 - w * 10  + 25
#this is the function we have to optimize - opt.minimize() function
def functionToMinimize():
	return w ** 2 - w * 10  + 25
def reset():	
    #set initial value for weight param (e.g. 0)
	w = tf.Variable(0.0)  
	return w

#First I reset x1 and x2 to (10, 10).
w = reset()
#Then choose the SGD(stochastic gradient descent) optimizer with rate = 0.1.
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
#Finally perform minimization using opt.minimize()with respect to the 
#function fu_minmizie without input values , since opt.minimize() would 
#refer to the provided var_listas the variables to be updated.
for i in range(50):
	print ('y = {:.1f}, w = {:.1f}'.format(calculateCost(w).numpy(), w.numpy()))
	opt.minimize(functionToMinimize, var_list=[w])


