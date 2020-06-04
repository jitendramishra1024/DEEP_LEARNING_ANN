import numpy as np
X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y = np.array([[0, 1, 0, 0]]).T
wij   = np.random.rand(3,1)
def cost(Y_true,Y_pred):
    MSE = np.square(np.subtract(Y_true,Y_pred)).mean()
    return MSE
def sigmoid(x, w):
    z = np.dot(x, w)
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(x, w):
    return sigmoid(x, w) * (1 - sigmoid(x, w))

def gradient_descent(x, y, iterations):
    np.random.seed(10) # for generating the same results
    wij   = np.random.rand(3,1)
    eta=0.01
    for i in range(iterations):
        print("iteration number",i)
        Xi = x
        yhat = sigmoid(Xi, wij)
        g_wij = np.dot(Xi.T, (y - yhat) * sigmoid_derivative(Xi, wij))
        wij += eta*g_wij
        mse=cost(y,yhat)
        print(mse)
    print('The final prediction from neural network are: ')
    print(yhat)
gradient_descent(X, y, 20000)

#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
#https://towardsdatascience.com/a-step-by-step-implementation-of-gradient-descent-and-backpropagation-d58bda486110