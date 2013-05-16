'''
Created on May 8, 2013

@author: priya
'''

from sklearn.multiclass import OneVsRestClassifier as ovr
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn import metrics
import numpy as np
import pydot
import StringIO
from sklearn import tree

trainfile="data/top4.csv"
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
def preprocess(trainfile):
    myMatrix=[]
    with open(trainfile,"r") as inputf:
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
            
            month=record[6].strip('\r\n')
            myMatrix.append([forum,crop,long(duration),village,float(age),month,topic])
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
        
        month=months.index(each[5])
        topic=topics.index(each[6])
        labels.append(topic)
        dataset.append([crop,dur,village,age,month])
    return dataset,labels


        
'''
Step.3 Training the model
'''
trainMat=[]
training=[]
trainMat=preprocess(trainfile)
training=prepare(trainMat)
labels=training[1]
#Decision Tree
#topicClf=dtc(criterion='entropy',random_state=0)
#topicClf=dtc(random_state=0)
#topicClf = RFC(n_estimators=12, max_features=5, random_state=0)
topicClf=ovr(dtc(criterion='entropy',random_state=0))
topicClf.fit(training[0],labels)
#print topicClf.multilabel_ 
#scores = cross_val_score(topicClf, training[0], labels)
#print "Mean2: ",scores.mean()    
                            
'''
topicClf = dtc(max_depth=None, min_samples_split=1,random_state=0)
scores = cross_val_score(topicClf, training[0], labels)
print "Mean1: ", scores.mean() 

                         

topicClf = ETC(n_estimators=10, max_depth=None,min_samples_split=1, random_state=0)
scores = cross_val_score(topicClf, training[0], labels)
print "Mean3: ",scores.mean()   
'''
#Step.4 Testing the model


testMat=[]
testing=[]
testData=[]
testMat=preprocess(testf)
testing=prepare(testMat)
testData=testing[0]
testLabels=testing[1]
predictions=topicClf.predict(testData)

val=topicClf.score(testData,testLabels)

pestEst=topicClf.estimators_[0]
disEst=topicClf.estimators_[1]
fertEst=topicClf.estimators_[2]
planEst=topicClf.estimators_[3]

pestProba=pestEst.predict_proba(testData)
print pestProba.shape
'''
for (x,y),val in np.ndenumerate(pestProba):
    print x,y,val
'''  
pestPred=pestEst.predict_proba(testData)
disPred=disEst.predict_proba(testData)
fertPred=fertEst.predict_proba(testData)
planPred=planEst.predict_proba(testData)
pestsornot=['Not','Pests']
disornot=['Not','Diseases']
fertornot=['Not','Fertilizers']
planornot=['Not','Crop Planning']
pp=[]
for p in pestPred:
    #pp.append(pestsornot[p])
    pp.append(p)

dp=[]
for p in disPred:
    #dp.append(disornot[p])
    dp.append(p)

fp=[]
for p in fertPred:
    #fp.append(fertornot[p])
    fp.append(p)

plp=[]
for p in planPred:
    #plp.append(planornot[p])
    plp.append(p)
preds=topicClf.predict(testData)
i=0
finals=[]
print pp
while(i<len(testData)):
    if((pp[i]-dp[i]-fp[i]-plp[i]).all()==0):
        finals.append("Other")
    else:
        finals.append(topics[preds[i]])
    i+=1
for f in finals:
    print f

print "Mean Accuracy: ",val
print "Classification Report: \n", metrics.classification_report(testLabels,predictions)
print topics

'''
import pydot
import StringIO
from sklearn import tree
dot_data = StringIO.StringIO() 
tree.export_graphviz(pestEst, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("AO.pdf") 



'''