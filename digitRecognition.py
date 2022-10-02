import numpy as np
import pandas as pd

######################################   Neural Network   ######################################
# Initialize parameters
def initParams():
    # Each picture has 784 pixels, 10 is the number of neurons in layer 1
    # Every neuron needs a weight for every input (pixel) -> 10x784
    W1 = np.random.randn(10, 784)
    # Every neuron has a bias, 10 neurons, 1 bias -> 10x1
    b1 = np.random.randn(10, 1)
    # Second layer has 10 neurons, inputs are outputs from the first layer
    # First layer gives 10 outputs -> second layer needs 10x10 weights
    W2 = np.random.randn(10, 10)
    # Every neuron has a bias, 10 neurons, 1 bias -> 10x1
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2

# Function to map layer outputs 
def ReLU(Z):
    # f(x) = 0 if x <= 0
    # f(x) = x if x > 0
    return np.maximum(0, Z)

# Map values according to softmax function
# Values are mapped to probabilities -> sum is 1
def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

# Forward propagation
def forwardProp(W1, b1, W2, b2, X):
    #print (W1.shape)
    #print (X.shape)
    #print(b1.shape)
    # Z1 is a vector-matrix dot product (+ bias) and represents outputs of the first layer
    Z1 = W1.dot(X) + b1
    #print(Z1.shape)
    A1 = ReLU(Z1)
    # Z2 is a vector-matrix dot product (+ bias) and represents outputs of the second layer
    # Z2 is the output of the neural network
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Convert labels to one hot encoding
def oneHot(Y):
    # Initialize a matrix of labels
    oneHotY = np.zeros((Y.size, Y.max() + 1))
    # Iterate trough every row and set the value to 1 on the position od the correct digit
    oneHotY[np.arange(Y.size), Y] = 1
    # Return transposed matrix
    return oneHotY.T

# Derivation of the ReLU function
def derivReLU(Z):
    # 0 if f(x) = 0
    # 1 if f(x) > 0
    return Z > 0

# Find out what weights and biases are responsible for the errors and how much each
# weight and bias contributed to the error
def backProp(Z1, A1, Z2, A2, W2, X, Y):
    # Number of pictures
    m = Y.size
    # Get one-hot labels
    oneHotY = oneHot(Y)
    # Calculate the error of the second layer
    dZ2 = A2 - oneHotY
    # How much W2 contributed to the error
    dW2 = 1 / m * dZ2.dot(A1.T)
    # How much b2 contributed to the error
    db2 = 1 / m * np.sum(dZ2, 1)
    # Calculate the error of the first layer
    dZ1 = W2.T.dot(dZ2) * derivReLU(Z1)
    # How much W2 contributed to the error
    dW1 = 1 / m * dZ1.dot(X.T)
    # How much b2 contributed to the error
    db1 = 1 / m * np.sum(dZ1, 1)
    return dW1, db1, dW2, db2

# Update parameters with alpha according to the error contributions calculated 
# alpha is a hyperparameter -> manualy picked, not trained
def updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    print(b1.shape)
    print(db1.shape)
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def getPredictions(A2):
    return np.argmax(A2, 0)

def getAccuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Get to the minimum of the loss function using gradient descent
def gradientDescent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = initParams()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backProp(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = updateParams(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        #if i % 10 == 0:
        print('Iteration: ', i)
        print('Accuracy: ', getAccuracy(getPredictions(A2), Y))

    return W1, b1, W2, b2
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

######################################        Main        ######################################
###################################### Data manipulation  ######################################
# Load data from csv
data = pd.read_csv('train.csv')
#print(data.head())

# Convert to nparray
data = np.array(data)

# Get dimensions
m, n = data.shape

# Shuffle and split into train and validate
np.random.shuffle(data)

dataVal = data[0:1000].T
valX = dataVal[1:n]
valY = dataVal[0]

dataTrain = data[1000:m].T
trainX = dataTrain[1:n]
trainY = dataTrain[0]

#print(valX[:, 0].shape)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

W1, b1, W2, b2 = gradientDescent(trainX, trainY, 100, 0.1)



