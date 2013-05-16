'''
Created on May 8, 2013

@author: priya
'''

from sklearn.tree import DecisionTreeClassifier as dtc
import cPickle
from sklearn.metrics import classification_report
from sklearn import cross_validation

#Input files required for classification
trainfile="data/TrainingSet.csv"
testf="data/TestSet.csv"

#Declaration of variables
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
crops=[]
topics=['Animal Husbandry', 'Crop Planning', 'Crop Variety', 'Diseases', 'Fertilizers/Bio-organic', 'Government', 'Harvesting', 'Horticulture', 'IPM Strategy', 'Irrigation', 'Land Preparation', 'Marketing', 'NGOs', 'Other', 'Pests', 'Seeds', 'Soil Care', 'Sowing', 'Weather', 'Weed Control']
testdata=[]

'''
Step:1 Pre-processing
The preprocess() function extracts the required fields from the .csv input file which has records in the following format:
crop, actual topic, call duration, month
Input: file containing the records
Output: a list containing the records
'''
def preprocess(inputfile):
    myMatrix=[]
    #messagecrop,messagetopic,listendurations,Month
    with open(inputfile,"r") as inputf:
        for row in inputf:
            record=row.split(",")
            crop=record[0]
            if crop=='':
                crop='None'
            topic=record[1]
            duration=record[2]
            if(crop not in crops):
                crops.append(crop)
            if(topic not in topics):
                topics.append(topic)
            month=record[3].strip('\r\n')
            myMatrix.append([crop,long(duration),month,topic])
    inputf.close()
    return myMatrix

'''
Step.2 Data Preparation
The prepare() function processes the list from the preprocess() function to separate the class labels and vectorize the nominal variables such as "crop" and "month" 
Input: list containing the records
Output: Two lists containing vectorized training data set and class labels respectively
'''
def prepare(myMatrix):
    labels=[]
    dataset=[]
    for each in myMatrix:
        crop=crops.index(each[0])
        dur=each[1]
        month=months.index(each[2])
        topic=topics.index(each[3])
        labels.append(topic)
        dataset.append([crop,dur,month])
    return dataset,labels

'''
Step.3 Training the model
Processed records are then passed to sklearn's Decision Tree to train the model.
'''

trainMat=[]
training=[]
trainMat=preprocess(trainfile)
training=prepare(trainMat)
labels=training[1]

'''
Fitting the training data into the decision tree
'''
topicClf = dtc(criterion='entropy',random_state=0)
topicClf.fit(training[0],labels)

'''
Cross validating the results using 10% of the training set as the test set 
'''
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
training[0], labels, test_size=0.1, random_state=0)
print "Cross Validation Score"
print topicClf.score(X_test, y_test)                                        


# Save the classifier
with open('topic_classifier.pkl', 'wb') as fid:
    cPickle.dump(topicClf, fid)    

'''
# load the classifier
with open('topic_classifier.pkl', 'rb') as fid:
    topicClf = cPickle.load(fid)
'''

'''
Step.4 Testing the model
Passing the test dataset to the classifier 
'''
testMat=[]
testing=[]
testData=[]
testMat=preprocess(testf)
testing=prepare(testMat)
testData=testing[0]
testLabels=testing[1]
predictions=topicClf.predict_proba(testData)
predictLabels=topicClf.predict(testData)
val=topicClf.score(testData,testLabels)
print "Mean Accuracy: ",val
print "Classification Report: \n", classification_report(testLabels, predictLabels)


'''
Formatting the output. Check output file: OutputClassifier
'''
with open(testf,'r') as testfile:
    for row in testfile:
        testdata.append(row)
print "Test Prediction Results"
print "The top row is the actual record: crop,actual topic, call duration, month."
print "The percentage-wise breakdown of the predicted topic are below each of the actual records."
testResult=[]
k=0
for each in predictions:
    print "Record #"+str(k)+" : "+str(testdata[k])
    j=0
    for i in each:
        if(i!=0):
            print [topics[j]+": "+str(round(i,3)*100)+"%"]
        j+=1
    k+=1
    print "**************************"