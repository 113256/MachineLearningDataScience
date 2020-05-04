from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from Modules import mnistHelper as mHelper
from Modules import deepnn as deepnn
from Modules import adamGradientDescent as adam
from Modules import testCases_v4a
(trainXOriginal, trainyOriginal), (testXOriginal, testyOriginal) = mnist.load_data()

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

#parameters = deepnn.L_layer_model(XTrain, yTrain,layers_dims = [XTrain.shape[0], 64, 10], num_iterations = 50, learning_rate = 0.8, lambd = 0.3)
parameters = deepnn.L_layer_model_mini(XTrain, yTrain,layers_dims = [XTrain.shape[0], 50, 10], num_iterations = 20, learning_rate = 0.8, lambd = 0, mini_batch_size=64, optimizer="gd", printIteration=2, updateIteration=2)
#parameters = deepnn.L_layer_model_mini(XTrain, yTrain,layers_dims = [XTrain.shape[0], 50, 10], num_iterations = 20, learning_rate = 0.8, lambd = 0, mini_batch_size=64, optimizer="adam",printIteration=2 ,updateIteration=2)
print ("On the training set:")
predictions_train = mHelper.predictMinst(XTrain, trainyOriginal, parameters)
print ("On the test set:")
predictions_test = mHelper.predictMinst(XTest, testyOriginal, parameters)
pred3 = mHelper.displayPrediction(np.atleast_2d(XTrain[:,200]).T,parameters)