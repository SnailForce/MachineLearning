from re import M
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio

def PCA_2D():
    data = spio.loadmat("data.mat")
    X = data['X']
    plotData(X, 'bo')

    # 去中心化
    X_norm, mu, sigma = featureNormalize(X)
    print(X.shape, X_norm.shape)
    plotData(X_norm, 'ro')
    # plt.show()

    # 协方差矩阵
    # S为特征值对角矩阵 以向量形式保存
    Sigma = np.dot(np.transpose(X_norm), X_norm) / (X.shape[0] - 1)
    U, S, V = np.linalg.svd(Sigma)
    print(Sigma.shape, U.shape, S.shape, V.shape)

    # TODO:
    plt.plot([mu[0], (mu +  U[:, 0])[0]],[mu[1], (mu + U[:, 0])[1]],'r--') 
    # plt.plot([mu[0], (mu + S[0] * U[:, 0])[0]],[mu[1], (mu + S[0] * U[:, 0])[1]],'g--') 
    plt.axis("square")
    plt.show()

    K = 1
    Z = projectData(X_norm, U, K)
    X_rec = recoverData(Z, U, K)

    plotData(X_norm, 'bo')
    plotData(X_rec, 'ro')
    plt.plot([0, U[0, 0]],[0, U[1, 0]],'g--') 
    for i in range(X_norm.shape[0]):
        plt.plot([X_norm[i, 0], X_rec[i, 0]],[X_norm[i, 1], X_rec[i, 1]],'k--') 
    plt.axis("square")
    plt.show()

def PCA_faceImage():
    data = spio.loadmat("data_faces.mat")
    X = data['X']
    print(X.shape)
    displayData(X[0 : 100])

    X_norm, mu, sigma = featureNormalize(X)
    Sigma = np.dot(np.transpose(X_norm), X_norm) / (X.shape[0] - 1)
    U, S, V = np.linalg.svd(Sigma)
    displayData(np.transpose(U[:, 0 : 36]))
    print(U.shape)

    K = 100
    Z = projectData(X_norm, U, K)
    print(Z.shape)
    X_rec = recoverData(Z, U, K)
    print(X_rec.shape)
    displayData(X_rec[0 : 100])

def featureNormalize(X):
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - mu[i]) / sigma[i]
    
    return X, mu, sigma

def plotData(X, marker):
    plt.plot(X[:, 0], X[:, 1], marker)

def projectData(X_norm, U, K):
    Z = np.zeros((X_norm.shape[0], K))
    U_reduce = U[:, 0 : K]
    Z = np.dot(X_norm, U_reduce)
    return Z

def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    U_recude = U[:, 0 : K]
    X_rec = np.dot(Z, np.transpose(U_recude))
    return X_rec

def displayData(X):
    sum = 0
    width = np.int32(np.round(np.sqrt(X.shape[1])))
    height = np.int32(X.shape[1] / width)
    rows_count = np.int32(np.floor(np.sqrt(X.shape[0])))
    cols_count = np.int32(np.ceil(X.shape[0] / rows_count))

    pad = 1
    display_array = np.ones((pad + rows_count * (height + pad), pad + cols_count * (width + pad)))

    for i in range(rows_count):
        for j in range(cols_count):
            max_val = np.max(np.abs(X[sum, :]))
            display_array[pad + i * (height + pad) : pad + i * (height + pad) + height, pad + j * (width + pad) : pad + j * (width + pad) + width] = X[sum, :].reshape(height, width, order='F') / max_val
            sum += 1
    
    plt.imshow(display_array, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    PCA_2D()
    PCA_faceImage()
