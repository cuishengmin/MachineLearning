#-*- coding: utf-8 -*-
from numpy import *
import operator


def createDataSet():
    characters=array([[1.0,1.1],[1.0,1.0],[1.0,1.2],[0,0],[0,0.1],[0,0.2]])
    labels=['A','A','A','B','B','B']
    return characters,labels

def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()


    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def main():
    sample=[0,0]
    k=5
    group,labels=createDataSet()
    label=classify(sample,group,labels,k)
    print("Classified Label:"+label)

if __name__=='__main__':
    main()
