import time
import numpy as np
import random
import math

def loadData(fileName):
    dataArr =[]
    labelArr = []
    
    f = open(fileName)

    for line in f.readlines():
        curLine = line.strip().split(',') # 去掉首尾空格或换行符，按逗号分隔
        dataArr.append([int(num) / 255 for num in curLine[1:]]) # 字符串转换为浮点型小数
        if int(curLine[0]) == 0:
            labelArr.append(1)
        else:
            labelArr.append(-1)
    return dataArr, labelArr


class SVM():
    def __init__(self, trainDataList, trainLabelList, sigma = 10, C = 200, epsilon = 0.001):
        self.trainDataMat = np.mat(trainDataList)
        self.trainLabelMat = np.mat(trainLabelList).T

        self.num = self.trainDataMat.shape[0]
        self.sigma = sigma
        self.C = C
        self.epsilon = epsilon

        self.kernel = self.calcKernel()
        self.b = 0
        self.alpha = np.zeros(self.num)
        
        self.E = [-self.trainLabelMat[i] for i in range(self.num)]

        self.supportVecIndex = []

    def calcSingleKernel(self, x1, x2):
        result = np.dot(x1 - x2, np.transpose(x1 - x2))
        result = -np.exp(-1 * result / (2 * self.sigma ** 2))
        return result
    
    def calcKernel(self):
        kernel = [[0 for _ in range(self.num)] for _ in range(self.num)]

        for i in range(self.num):
            X = self.trainDataMat[i]
            for j in range(i, self.num):
                Z = self.trainDataMat[j]
                result = np.dot(X - Z, np.transpose(X - Z)) 
                result = np.exp(-1 * result / (2 * self.sigma ** 2))
                kernel[i][j] = result
                kernel[j][i] = result
        return kernel
   
    def isSatisfyKKT(self, i):
        gxi = self.calcGxi(i)
        yi = self.trainLabelMat[i]

        if self.alpha[i] == 0:
            return yi * gxi >= 1
        elif 0 < self.alpha[i] < self.C:
            return yi * gxi == 1
        else:
            return yi * gxi <= 1

        # #依据7.111
        # if (math.fabs(self.alpha[i]) < self.epsilon) and (yi * gxi >= 1):
        #     return True
        # #依据7.113
        # elif (math.fabs(self.alpha[i] - self.C) < self.epsilon) and (yi * gxi <= 1):
        #     return True
        # #依据7.112
        # elif (self.alpha[i] > -self.epsilon) and (self.alpha[i] < (self.C + self.epsilon)) \
        #         and (math.fabs(yi * gxi - 1) < self.epsilon):
        #     return True

        # return False

    def calcGxi(self, i):
        gxi = self.b
        index = [j for j, alpha in enumerate(self.alpha) if alpha != 0]
        for j in index:
            gxi += self.alpha[j] * self.trainLabelMat[j] * self.kernel[i][j]
        return gxi

    def calcE(self, i):
        return self.calcGxi(i) - self.trainLabelMat[i]

    def getAlpha(self):
        index_list = [i for i in range(self.num) if 0 < self.alpha[i] < self.C]
        non_satisft_list = [i for i in range(self.num) if i not in index_list]
        index_list.extend(non_satisft_list)

        for i in index_list:
            if self.isSatisfyKKT(i):
                continue
            E1 = self.calcE(i)
            if E1 >= 0:
                j = min(range(self.num), key=lambda x : self.calcE(x))
            else:
                j = max(range(self.num), key=lambda x : self.calcE(x))
            return i, j
    
    def compareLH(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L: 
            return L
        else:
            return alpha

    def train(self, iter = 100):
        iterStep = 0

        while iterStep < iter:
            print('iter: %d:%d' % (iterStep, iter))
            iterStep += 1

            i, j = self.getAlpha()
            
            alphaOld_1 = self.alpha[i]
            alphaOld_2 = self.alpha[j]
            if self.trainLabelMat[i] == self.trainLabelMat[j]:
                L = max(0, alphaOld_2 + alphaOld_1 - self.C)
                H = min(self.C, alphaOld_1 + alphaOld_2)
            else:
                L = max(0, alphaOld_2 - alphaOld_1)
                H = min(self.C, self.C + alphaOld_2 - alphaOld_1)
            if L == H:
                continue
            E1 = self.calcE(i)
            E2 = self.calcE(j)
            alphaNew_2 = self.compareLH(alphaOld_2 + self.trainLabelMat[j] * (E1 - E2) / (self.kernel[i][i] + self.kernel[j][j] - 2 * self.kernel[i][j]), L, H)
            alphaNew_1 = alphaOld_1 + self.trainLabelMat[i] * self.trainLabelMat[j] * (alphaOld_2 - alphaNew_2)
            bNew_1 = -E1 - self.trainLabelMat[i] * self.kernel[i][i] * (alphaNew_1 - alphaOld_1) - self.trainLabelMat[j] * self.kernel[j][i] * (alphaNew_2 - alphaOld_2) + self.b
            bNew_2 = -E2 - self.trainLabelMat[i] * self.kernel[i][j] * (alphaNew_1 - alphaOld_1) - self.trainLabelMat[j] * self.kernel[j][j] * (alphaNew_2 - alphaOld_2) + self.b
            if 0 < alphaNew_1 < self.C:
                bNew = bNew_1
            elif 0 < alphaNew_2 < self.C:
                bNew = bNew_2
            else:
                bNew = (bNew_1 + bNew_2) / 2
            self.alpha[i] = alphaNew_1
            self.alpha[j] = alphaNew_2
            self.b = bNew

            self.E[i] = self.calcE(i)
            self.E[j] = self.calcE(j)

        for i in range(self.num):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)  
    
    def predict(self, x):
        result = self.b
        for i in self.supportVecIndex:
            result += self.alpha[i] * self.trainLabelMat[i] * self.calcSingleKernel(self.trainDataMat[i], np.mat(x))
        return np.sign(result)

    def test(self, testDataList, testLabelList):
        cnt = 0
        print(len(testDataList))
        for i in range(len(testDataList)):
            # print(self.predict(testDataList[i]), testLabelList[i])
            if self.predict(testDataList[i]) == testLabelList[i]:        
                cnt += 1
        return cnt / len(testDataList)           

if __name__ == '__main__':
    start = time.time()

    trainDataList, trainLabelList = loadData('./mnist_train.csv')
    testDataList, testLabelList = loadData('./mnist_test.csv')

    svm = SVM(trainDataList[:1000], trainLabelList[:1000], 10, 0.1, 0.001)
    svm.train()
    accuracy = svm.test(testDataList[:100], testLabelList[:100])
    print("Accuracy: %d%%" % (accuracy * 100))

    print("time span: ", time.time() - start)