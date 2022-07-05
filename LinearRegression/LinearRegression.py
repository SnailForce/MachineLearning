import numpy as np
from matplotlib import pyplot as plt


def linearRegression(alpha=0.01, num_iters=400):
    print("load data...\n")

    data = np.loadtxt("data.txt", delimiter=",", dtype=np.float64)

    X = data[:, 0:-1]
    y = data[:, -1]
    m = len(y)
    col = data.shape[1]

    X, mu, sigma = featureNormalize(X)
    plot_X1_X2(X)

    X = np.hstack((np.ones((m, 1)), X))

    print("gradient descent...\n")

    theta = np.zeros((col, 1))
    y = y.reshape(-1, 1)

    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

    plotJ(J_history, num_iters)

    return mu, sigma, theta


def featureNormalize(X):
    X_norm = np.array(X)

    mu = np.mean(X_norm, 0)
    sigma = np.std(X_norm, 0)

    for i in range(X.shape[1]):
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]

    return X_norm, mu, sigma


def plot_X1_X2(X):
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)

    tmp_theta = np.matrix(np.zeros((n, num_iters)))

    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = np.dot(X, theta)
        tmp_theta[:, i] = theta - (alpha / m) * \
            np.dot(np.transpose(X), np.dot(X, theta) - y)
        theta = tmp_theta[:, i]
        J_history[i] = computeCost(X, y, theta)
        print('.', end=' ')

    return theta, J_history


def computeCost(X, y, theta):
    m = len(y)
    J = 0

    J = np.dot(np.transpose(X * theta - y), X * theta - y) / (2 * m)
    return J

def plotJ(J_history, num_iters):
    x = np.arange(1, num_iters + 1)
    plt.plot(x, J_history)
    plt.xlabel("iters")
    plt.ylabel("J")
    plt.title("curve")
    plt.show()


def testLinearRegression():
    mu, sigma, theta = linearRegression(0.01, 400)
    print(mu, sigma, theta)


if __name__ == "__main__":
    # testLinearRegression()
    linearRegression()
