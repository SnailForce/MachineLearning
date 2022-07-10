import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from scipy import optimize


def logisticRegressionOneVsAll():
    data = spio.loadmat("data_digits.mat")
    X = data["X"]
    y = data["y"]
    print(X.shape, y.shape)
    num_labels = 10

    rand_indices = [np.random.randint(0, X.shape[0]) for x in range(100)]
    displayData(X[rand_indices])

    Lambda = 0.1
    all_theta = OneVsAll(X, y, num_labels, Lambda)

    p = predict(X, all_theta)
    print("Accuracy=%f%%"%np.mean(np.float64(p == y) * 100))


def displayData(imgData):
    sum = 0
    pad = 1
    display_array = -np.ones((pad + 10 * (20 + pad), pad + 10 * (20 + pad)))
    for i in range(10):
        for j in range(10):
            display_array[
                pad + i * (20 + pad) : pad + i * (20 + pad) + 20,
                pad + j * (20 + pad) : pad + j * (20 + pad) + 20,
            ] = imgData[sum, :].reshape(
                20, 20, order="F"
            )  # order=F指定以列优先，在matlab中是这样的，python中需要指定，默认以行
            sum += 1

    plt.imshow(display_array, cmap='gray')
    plt.axis('off')
    plt.show()

def OneVsAll(X, y, num_labels, Lambda):
    all_theta = np.zeros((X.shape[1] + 1, num_labels))
    initial_theta = np.zeros((X.shape[1] + 1, 1))
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    class_y = np.zeros((X.shape[0], num_labels))
    
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(-1)
        result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X, class_y[:, i], Lambda))
        all_theta[:, i] = result
    
    return all_theta

def costFunction(initial_theta, X, y, initial_lambda):
    J = 0

    h = sigmoid(np.dot(X, initial_theta))
    tmp_theta = initial_theta.copy()
    tmp_theta[0] = 0

    tmp = np.dot(np.transpose(tmp_theta), tmp_theta)
    J = (-np.dot(np.transpose(y), np.log(h)) -
         np.dot(np.transpose(1-y), np.log(1-h)) + initial_lambda / 2 * tmp) / len(y)
    return J

def sigmoid(z):
    h = np.zeros((len(z), 1))

    h = 1.0 / (1.0 + np.exp(-z))
    return h

def gradient(initial_theta, X, y, initial_lambda):
    grad = np.zeros((initial_theta.shape[0]))

    h = sigmoid(np.dot(X, initial_theta))
    tmp_theta = initial_theta.copy()
    tmp_theta[0] = 0

    grad = (np.dot(np.transpose(X), h-y) + initial_lambda * tmp_theta) / len(y)
    return grad

def predict(X, theta):
    p = np.zeros((X.shape[0], 1))
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    h = sigmoid(np.dot(X, theta))

    for i in range(X.shape[0]):
        p[i] = np.array(np.where(h[i, :] == np.max(h, axis=1)[i]))
    return p

def testLogisticRegressionOneVsAll():
    logisticRegressionOneVsAll()

if __name__ == "__main__":
    testLogisticRegressionOneVsAll()
