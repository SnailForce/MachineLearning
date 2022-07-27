from sklearn.svm import SVC

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

clf = SVC(C=1.0, kernel='rbf', gamma=0.1)

trainDataList, trainLabelList = loadData('./mnist_train.csv')
testDataList, testLabelList = loadData('./mnist_test.csv')

clf.fit(trainDataList[:1000], trainLabelList[:1000])
print(clf.score(testDataList[:100], testLabelList[:100]))


