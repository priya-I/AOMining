'''
Created on May 9, 2013

@author: priya
'''
print __doc__

from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import numpy as np

inputfile="data/top4.csv"
testf="data/testdata.csv"
villages=[]
trtgrp=['O','T','TH']
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
forums=['Questions and Answers','Announcements','Experience Sharing','Radio']
crops=[]
topics=[]

def preprocess(inputfile):
    myMatrix=[]
    with open(inputfile,"r") as inputf:
        for row in inputf:
            record=row.split(",")
            forum=record[0]
            crop=record[1]
            if crop=='':
                crop='None'
            topic=record[2]
            duration=record[3]
            village=record[4]
            if(village not in villages):
                villages.append(village)
            if(crop not in crops):
                crops.append(crop)
            if(topic not in topics):
                topics.append(topic)
            age=record[5]
            trt=record[6]
            month=record[7].strip('\r\n')
            myMatrix.append([forum,crop,long(duration),village,float(age),trt,month,topic])
    inputf.close()
    return myMatrix

def prepare(myMatrix):
    labels=[]
    dataset=[]
    for each in myMatrix:
        forum=forums.index(each[0])
        crop=crops.index(each[1])
        dur=each[2]
        village=villages.index(each[3])
        age=each[4]
        trt=trtgrp.index(each[5])
        month=months.index(each[6])
        topic=topics.index(each[7])
        labels.append(topic)
        dataset.append([forum,crop,dur,village,age,trt,month])
    return dataset,labels


trainMat=[]
training=[]
trainMat=preprocess(inputfile)
training=prepare(trainMat)
# Load the digits dataset
#digits = load_digits()
X = training[0]
y = training[1]
X=np.asarray(X)
y=np.asarray(y)
trainset= X.reshape((len(X), -1))
labels=y
print trainset.shape
print labels.shape
# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", C=1)
rfe = RFE(estimator=svc, n_features_to_select=4, step=4)
rfe.fit(trainset, labels)
ranking = rfe.ranking_.reshape(trainset[0].shape)
#print ranking
'''
# Plot pixel ranking
import pylab as pl
pl.matshow(ranking)
pl.colorbar()
pl.title("Ranking of pixels with RFE")
pl.show()
'''