import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#-------------------------------------preProcessing--------------------------------
data1 = pd.read_csv('mnist_train.csv')
data = np.array(data1)
m, n = data.shape
data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

data2 = pd.read_csv('mnist_test.csv')
data = np.array(data2)
m, n = data.shape
data_test = data.T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.


#----------------------------------functions---------------------------------------

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def loss_func(y_pred, y_true):
    y_true = one_hot(y_true)
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred))
    return loss


def backward_prop(Z1, A1, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate):
    W1 = W1 - learningRate * dW1
    b1 = b1 - learningRate * db1    
    W2 = W2 - learningRate * dW2  
    b2 = b2 - learningRate * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return(np.sum(predictions == Y) / Y.size)

def gradient_descent(X, Y, learningRate, epochs):
    W1, b1, W2, b2 = init_params()
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learningRate)
        if i % 100 == 0:
            print("Epoch: ", i)
            predictions = get_predictions(A2)
            print("accuracy:", get_accuracy(predictions, Y))
            print("loss:", loss_func(A2, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_single(index, W1, b1, W2, b2):
    current_image = X_test[:, index, None]
    prediction = make_predictions(X_test[:, index, None], W1, b1, W2, b2)
    label = Y_test[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image)
    plt.show()

#-------------------------------training------------------------------------------------

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 2000)
print("--------------------")
#------------------------------testing-----------------------------------------------
test_predictions = make_predictions(X_test, W1, b1, W2, b2)
print("Accuracy on test dataset:")
print(get_accuracy(test_predictions, Y_test))
print("--------------------------")
#---------------------------------testing on a single image---------------------------------------------------

print("On a single image:")
test_single(666, W1, b1, W2, b2)
