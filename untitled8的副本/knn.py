#         直接调包

# from sklearn import neighbors
# from sklearn import datasets
#
# knn = neighbors.KNeighborsClassifier()
#
# iris = datasets.load_iris()
# knn.fit(iris.data,iris.target)
# predict_lable = knn.predict([[0.1,0.2,0.3,0.4]])
# print(predict_lable)



#            实现 knn
import csv
import math
import random
import operator

def loadfile(file,split,trainingSet = [],testSet = []):
    with open(file,'rt') as datafile:
        data = csv.reader(datafile)
        dataset = list(data)
        for x in range(1,len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
def eduli(instance1,instance2,instance_lenth):
    distance = 0
    for x in range(instance_lenth ):
        distance += pow(instance1[x] -instance2[x],2)
    return math.sqrt(distance)

def neibor_instance(trainingset,test_instance,k):
    distance_list = []
    lenth = len(test_instance) - 1
    for x in range(len(trainingset)):
        dis = eduli(trainingset[x],test_instance,lenth)
        distance_list.append((dis,trainingset[x]))
    distance_list.sort(key=operator.itemgetter(0))
    neighbors = []
    for x in range(k):
        neighbors.append(distance_list[x][1])
    return neighbors
def selct_first_K_instance(neihbor):
    categorylist = {}
    flag_traning = []
    for x in range(len(neihbor)):
        class_label = neihbor[x][-1]
        if class_label not in flag_traning:
            categorylist[class_label] = 1
            flag_traning.append(class_label)
        else :
            categorylist[class_label] += 1
    sorted_category = sorted(categorylist.items(),key=operator.itemgetter(1),reverse=True)   #dict.items()返回的是一个完整的列表，而dict.iteritems()返回的是一个生成器(迭代器)
    return sorted_category[0][0]
def calculate_accurancy(testSet,predicted_results):
    count = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predicted_results[x]:
            count += 1
    return count/len(testSet) * 100.0
def main():
    trainingSet = []
    testSet = []
    path = r'/Users/wss/Desktop/iris.csv'
    split = 0.67
    loadfile(path,split,trainingSet,testSet)
    print('trainingSet: ' + repr(len(trainingSet)))
    print('testSet : '+ repr(len(testSet)))
    k = 3

    preditions = []
    for x in range(len(testSet)):
        neibor = neibor_instance(trainingSet, testSet[x], k)
        sorted_categories = selct_first_K_instance(neibor)
        preditions.append(sorted_categories)
        print('-->> predict:'+sorted_categories + '-->> actual :' +testSet[x][-1])
    accurancy = calculate_accurancy(testSet,preditions)
    print(accurancy)
main()











