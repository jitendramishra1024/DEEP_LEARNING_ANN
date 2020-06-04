import numpy as np
X = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
y = np.array([[0, 1, 0, 0]]).T
#x is  4 X 3 matrix weight wij=3 X4  output of hidden layer will be 4 X 4 matrix 
#output of hidden layer 4X4 wjk will be 4 X1 so out put will be 4X1 same as Y
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
    wij   = np.random.rand(3,4)
    wjk   = np.random.rand(4,1)
    eta=1
    for i in range(iterations):
        print("iteration number",i)
        Xi = x
        Xj = sigmoid(Xi, wij)
        yhat = sigmoid(Xj, wjk)
        # gradients for hidden to output weights
        g_wjk = np.dot(Xj.T, (y - yhat) * sigmoid_derivative(Xj, wjk))
        # gradients for input to hidden weights
        g_wij = np.dot(Xi.T, np.dot((y - yhat) * sigmoid_derivative(Xj, wjk), wjk.T) * sigmoid_derivative(Xi, wij))
        wij += g_wij
        wjk += g_wjk
        mse=cost(y,yhat)
        print(mse)
    print('The final prediction from neural network are: ')
    print(np.around(yhat,decimals=4))
gradient_descent(X, y, 1500)

#https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
#https://towardsdatascience.com/a-step-by-step-implementation-of-gradient-descent-and-backpropagation-d58bda486110