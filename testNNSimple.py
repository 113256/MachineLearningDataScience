from deepnn import *


#linear_activation_forward_test case" - PASS
A_prev, W, b = linear_activation_forward_test_case()
A, linear_activation_cache = linear_forward_Activation(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))
A, linear_activation_cache = linear_forward_Activation(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))

#L-MODEL FORWARD-  PASS
X, parameters = L_model_forward_test_case_2hidden()
#print(parameters)
AL, caches = L_Model_Forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


dAL, linear_activation_cache = linear_activation_backward_test_case()
print(linear_activation_cache)

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid", lambd = 0)
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu", lambd= 0)
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

print("L MODEL BACK TEST ")
#L MODEL BACKWARD
AL, Y_assess, caches = L_model_backward_test_case()
#print(caches)
grads = L_Model_Backwards(AL, Y_assess, caches, 0)
print_grads(grads)




train_X, train_Y, test_X, test_Y = load_2D_dataset()

parameters = L_layer_model(train_X, train_Y,layers_dims = [train_X.shape[0], 20, 3, 1], num_iterations = 25000, learning_rate = 0.3, lambd = 0.3)
#parameters = L_layer_model_mini(train_X, train_Y,layers_dims = [train_X.shape[0], 20, 3, 1], num_iterations = 20000, learning_rate = 0.3, lambd = 0, mini_batch_size=64, optimizer="gd", printIteration=100, updateIteration=100)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)


plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)# -*- coding: utf-8 -*-

plt.title("Model without regularization (Test Set)")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), test_X, test_Y)# -*- coding: utf-8 -*-
