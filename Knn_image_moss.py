import numpy as np
import pandas as pd
import glob
import random
import cv2
import sys
import os
import scipy
import operator
import scipy.spatial
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time
import matplotlib.pyplot as plt
import warnings
from itertools import repeat
warnings.simplefilter(action='ignore', category=FutureWarning)

def distance(a, b):
    dist = scipy.spatial.distance.euclidean(a, b)
    return dist


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = distance(testInstance, trainingSet.iloc[x,])
        distances.append((trainingSet.iloc[x,], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def knnPredict(neighbors):
    catVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in catVotes:
            catVotes[response] += 1
        else:
            catVotes[response] = 1
    sortedVotes = sorted(catVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def runTest(trainingSet, testSet, k):
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = knnPredict(neighbors)
        predictions = []
        predictions.append(result)
        return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(0,len(testSet)):
        if testSet.iloc[x,-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

def findImages(dirpath):
    os.chdir(dirpath)
    headPat = "**/people/*.jpg"
    dogPat = "**/dogs/*.jpg"
    headList = []
    dogList = []
    for x in glob.glob(dogPat):
        dogList.append(os.path.join(dirpath, x))
    for x in glob.glob(headPat):
        headList.append(os.path.join(dirpath, x))

    return headList, dogList

def extractFeatures(headList,dogList):
    ColNames = ["Blue", "Green", "Red", "Class"]
    headFeatures = pd.DataFrame(columns=ColNames)
    dogFeatures = pd.DataFrame(columns=ColNames)


    rTemp = []
    gTemp = []
    bTemp = []

    for z in range(0,len(dogList)):
        temp = cv2.imread(dogList[z])
        for x in range(0,temp.shape[0]):
            for y in range(0,temp.shape[1]):
                bTemp.append(temp.item(x,y,0))
                gTemp.append(temp.item(x,y,1))
                rTemp.append(temp.item(x,y,2))
        dogFeatures = dogFeatures.append({'Blue': np.mean(bTemp), 'Green': np.mean(gTemp), 'Red':np.mean(rTemp), 'Class':0}, ignore_index=True)
        rTemp.clear()
        gTemp.clear()
        bTemp.clear()
    for z in range(0, len(headList)):
        temp = cv2.imread(headList[z])
        for x in range(0, temp.shape[0]):
            for y in range(0, temp.shape[1]):
                bTemp.append(temp.item(x, y, 0))
                gTemp.append(temp.item(x, y, 1))
                rTemp.append(temp.item(x, y, 2))
        headFeatures = headFeatures.append({'Blue': np.mean(bTemp), 'Green': np.mean(gTemp), 'Red':np.mean(rTemp), 'Class':1}, ignore_index=True)
        rTemp.clear()
        gTemp.clear()
        bTemp.clear()

    return headFeatures, dogFeatures


def shuffle(imgList):
    a = len(imgList)
    b = a-1
    for d in range(b,0,-1):
        e=random.randint(0,d)
        if e == d:
            continue
        imgList[d], imgList[e] = imgList[e], imgList[d]
    return imgList

def KNNpartition(headList, dogList):
    partList = []
    length = len(headList)
    baseLen = length//5

    head1 = headList.iloc[:baseLen]
    head2 = headList.iloc[baseLen:(2*baseLen)]
    head3 = headList.iloc[(2*baseLen):(3*baseLen)]
    head4 = headList.iloc[(3*baseLen):(4*baseLen)]
    head5 = headList.iloc[(baseLen*4):]

    dog1 = dogList.iloc[:baseLen]
    dog2 = dogList.iloc[baseLen:(2*baseLen)]
    dog3 = dogList.iloc[(2*baseLen):(3*baseLen)]
    dog4 = dogList.iloc[(3*baseLen):(4*baseLen)]
    dog5 = dogList.iloc[(baseLen*4):]

    part1 = dog1.append(head1)
    part1 = part1.reset_index(drop=True)
    part2 = dog2.append(head2)
    part2 = part2.reset_index(drop=True)
    part3 = dog3.append(head3)
    part3 = part3.reset_index(drop=True)
    part4 = dog4.append(head4)
    part4 = part4.reset_index(drop=True)
    part5 = dog5.append(head5)
    part5 = part5.reset_index(drop=True)

    partList = [part1,part2,part3,part4,part5]

    return partList

def kfold(partList):
    start = time.perf_counter()
    k_experiment = []
    print("KNN")
    for nearest in range(1,11):
        print("")
        print("k = " + str(nearest))
        temp = []
        for k in range(0,5):
            valSet = partList[k]
            tiered_trSet = partList[~k]
            trSet = []
            for sublist in tiered_trSet:
                for item in sublist:
                    trSet.append(item)
            trSet = pd.DataFrame(trSet)
            valSet = pd.DataFrame(valSet)


            trSet = trSet.reset_index(drop=True)
            valSet = valSet.reset_index(drop=True)
            votes = []
            for z in range(0,len(valSet)):
                neighbors = getNeighbors(trSet, valSet.iloc[z,:], nearest)
                votes.append(knnPredict(neighbors))
            temp.append(getAccuracy(valSet,votes))
            print("accuracy of fold " + str((k)%10+1)+ ": " + str(getAccuracy(valSet,votes)))
            votes.clear()
        tempo = np.mean(temp)
        sd = np.std(temp)
        tempos = [nearest, tempo]
        k_experiment.append(tempos)
        print("average accuracy: " + str(np.mean(temp)))
        print("Standard Deviation: " + str(sd))
        temp.clear()
    print()
    print("running time: " + str(time.perf_counter()-start))
    k_experiment = pd.DataFrame(k_experiment)
    k_experiment.columns = ["K value", "Average Accuracy"]
    print(k_experiment)
    k_experiment.plot(x='K value', y='Average Accuracy', kind='bar')
    plt.ylim(40, 102)
    print(
        "Outputting Bar Graph with the average k-fold cross validated accuracy for the k-nearest neighbors algorithm over different values of k")
    plt.savefig("k_experiment.png")
    print("saving figure")
    plt.clf()
    return np.mean(k_experiment['Average Accuracy']), max(k_experiment['Average Accuracy'])

def naiveBayes(partList):
    print("Naive Bayes")
    temp = []
    for k in range(0,5):
        valSet = partList[k]
        tiered_trSet = partList[~k]
        trSet = []
        for sublist in tiered_trSet:
            for item in sublist:
                trSet.append(item)



        start = time.perf_counter()
        classifier = GaussianNB()
        trainX = trSet.iloc[:,:-1]
        trainY = trSet.iloc[:, -1]
        testX = valSet.iloc[:,:-1]
        testY = valSet.iloc[:,-1]
        classifier.fit(trainX, trainY)
        accuracy = float(classifier.score(testX,testY))*100
        print("Accuracy of fold " + str(k+1) + ": " + str(accuracy))
        temp.append(accuracy)
    print("average accuracy: " + str(np.mean(temp)))

    print()
    print("running time: " + str(time.perf_counter() - start))
    return np.mean(temp)

def svmClassifier(partList):
    print("")
    temp = []
    for k in range(0,5):
        valSet = partList[k]
        tiered_trSet = partList[~k]
        trSet = []
        for sublist in tiered_trSet:
            for item in sublist:
                trSet.append(item)
        trSet = pd.DataFrame(trSet)
        valSet = pd.DataFrame(valSet)

        trSet = trSet.reset_index(drop=True)
        valSet = valSet.reset_index(drop=True)
        start = time.perf_counter()
        classifier = SVC()
        trainX = trSet.iloc[:, :-1]
        trainY = trSet.iloc[:, -1]
        testX = valSet.iloc[:,:-1]
        testY = valSet.iloc[:,-1]
        classifier.fit(trainX, trainY)
        accuracy = float(classifier.score(testX,testY))*100
        print("Accuracy of fold " + str(k+1) + ": " + str(accuracy))
        temp.append(accuracy)
    print("average accuracy: " + str(np.mean(temp)))

    print()
    print("running time: " + str(time.perf_counter() - start))
    return np.mean(temp)

def run (filePath):
    headList, dogList = findImages(filePath)
    shuffle(headList)
    shuffle(dogList)
    headFeatures, dogFeatures = extractFeatures(headList, dogList)
    partList = KNNpartition(headFeatures,dogFeatures)
    kNNacc, maxKNNacc = kfold(partList)
    print("")
    nbAcc = naiveBayes(partList)
    print("")
    svmAcc = svmClassifier(partList)
    comparison = [maxKNNacc,nbAcc,svmAcc]
    colNames = ["KNN", "Naive Bayes", "Support Vector Machine"]
    title = ["Percent Accuracy"]
    compFrame = pd.DataFrame(comparison,colNames, columns=title)
    print(compFrame)
    plt.show(compPlot)


run(sys.argv[1])