import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def logisticRegression():
    data = np.loadtxt("data1.txt", delimiter=',', dtype=np.float64)
    X = data[:, :-1]
    y = data[:, -1]
    plot_data(X, y)

    X = mapFeature(X[:, 0], X[:, 1])
    initial_theta = np.zeros((X.shape[1], 1))
    initial_lambda = 0.1

    J = costFunction(initial_theta, X, y, initial_lambda)
    print(J)

    '''调用scipy中的优化算法fmin_bfgs 拟牛顿法Broyden-Fletcher-Goldfarb-Shanno
    - costFunction是自己实现的一个求代价的函数
    - initial_theta表示初始化的值
    - fprime指定costFunction的梯度
    - args是其余测参数 以元组的形式传入 最后会将最小化costFunction的theta返回 
    '''
    result = optimize.fmin_bfgs(
        costFunction, initial_theta, fprime=gradient, args=(X, y, initial_lambda))
    p = predict(X, result)
    print("Accuracy=%f%%" % np.mean(np.float64(p == y) * 100))

    X = data[:, 0:-1]
    y = data[:, -1]
    plotDecisionBoundary(result, X, y)


def plot_data(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')
    plt.plot(X[neg, 0], X[neg, 1], 'bo')
    plt.show()


# 55% -> 85%
def mapFeature(X1, X2):
    degree = 2
    out = np.ones((X1.shape[0], 1))  # 加入一列全1

    for i in range(1, degree + 1):  # 1 2
        for j in range(i + 1):  # 2 3
            tmp = (X1 ** (i - j)) * (X2 ** j)
            out = np.hstack((out, tmp.reshape(-1, 1)))

    return out


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
    p = sigmoid(np.dot(X, theta))
    for i in range(X.shape[0]):
        p[i] = 1 if p[i] > 0.5 else 0
    return p


def plotDecisionBoundary(theta, X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.figure(figsize=(15, 12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')
    plt.plot(X[neg, 0], X[neg, 1], 'bo')
    plt.title("Decision Boundary")

    # 根据数据设置
    u = np.linspace(30, 100, 50)
    v = np.linspace(30, 100, 50)

    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = np.dot(mapFeature(u[i].reshape(
                1, -1), v[j].reshape(1, -1)), theta)

    z = np.transpose(z)
    plt.contour(u, v, z, [0, 0.01])
    plt.show()


def testLogisticRegression():
    logisticRegression()


if __name__ == "__main__":
    testLogisticRegression()
