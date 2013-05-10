'''
Created on May 8, 2013

@author: priya
'''

from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import metrics
from sklearn import tree


inputfile="data/top4.csv"
testf="data/testdata.csv"
villages=[]
trtgrp=['O','T','TH']
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
forums=['Questions and Answers','Announcements','Experience Sharing','Radio']
crops=[]
topics=[]
#forum,crops,topic,dur,villag,age,group,month

'''
Step:1 Pre-processing
'''
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

'''
Step.2 Data Preparation
'''
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


'''
Step.3 Training the model
'''
trainMat=[]
training=[]
trainMat=preprocess(inputfile)
training=prepare(trainMat)
labels=training[1]
topicClf=dtc(criterion='entropy',random_state=0)
topicClf.fit(training[0],labels)

'''
Step.4 Testing the model
'''
testMat=[]
testing=[]
testData=[]
testMat=preprocess(testf)
testing=prepare(testMat)
testData=testing[0]
testLabels=testing[1]
predictions=topicClf.predict(testData)
for p in predictions:
    print topics[p]
val=topicClf.score(testData,testLabels)
print "Mean Accuracy: ",val
print "Classification Report: \n", metrics.classification_report(testLabels,predictions)
print topics


