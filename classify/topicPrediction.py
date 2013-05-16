'''
Created on May 8, 2013
@author: priya

Decision Tree Classifier for Topic Prediction of the Avaaj Otalo Platform
'''

from sklearn.tree import DecisionTreeClassifier as dtc
import cPickle
from sklearn.metrics import classification_report
from sklearn import cross_validation


'''
Step:1 Pre-processing
The preprocess() function extracts the required fields from the .csv input file which has records in the following format:
crop, actual topic, call duration, month
Input: file containing the records
Output: a list containing the records
'''
def old_preprocess(inputfile):
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
        if choice==1:
            topic=topics.index(each[3])
            labels.append(topic)
        dataset.append([crop,dur,month])
    return dataset,labels

'''
Step.3 Training the model
Processed records are then passed to sklearn's Decision Tree to train the model.
'''
def trainClassifier():
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
    
    
    #Cross validating the results using 10% of the training set as the test set 
    '''
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    training[0], labels, test_size=0.1, random_state=0)
    print "Cross Validation Score"
    print topicClf.score(X_test, y_test)                                        
    '''
    '''
    # Save the classifier
    with open('topic_classifier.pkl', 'wb') as fid:
        cPickle.dump(topicClf, fid)    
    '''


'''
Step.4 Testing the model
Passing the test dataset to the classifier 
'''
def testClassify():
    # load the classifier
    with open('topic_classifier.pkl', 'rb') as fid:
        topicClf = cPickle.load(fid)

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
    return predictions

'''
Formatting the output. Check output file: data/OutputClassifier
'''
def outputResults(predictions,testrows):
    for row in testrows:
        testdata.append(row)    
    print "Test Prediction Results"
    print "The top row is the actual record: crop,actual topic, call duration, month."
    print "The percentage-wise breakdown of the predicted topic are below each of the actual records."
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

if __name__ == '__main__':
    #Input files required for classification
    trainfile="data/TrainingSet.csv"
    testf="data/TestSet.csv"
    print "** Topic Predictor for the Avaaj Otalo Platform **"
    #Declaration of variables
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    crops=['Cotton', 'None', 'Cumin', 'Mustard', 'Other', 'Wheat', 'Castor', 'Groundnut', 'Onion', 'Sorghum', 'Brinjal', 'Chilli', 'Gram', 'Paddy', 'Millet', 'Sesame', 'Banana', 'Maize', 'Garlic', 'Tobacco', 'Papaya']
    topics=['Animal Husbandry', 'Crop Planning', 'Crop Variety', 'Diseases', 'Fertilizers/Bio-organic', 'Government', 'Harvesting', 'Horticulture', 'IPM Strategy', 'Irrigation', 'Land Preparation', 'Marketing', 'NGOs', 'Other', 'Pests', 'Seeds', 'Soil Care', 'Sowing', 'Weather', 'Weed Control']
    testdata=[]

    choice=input("Enter 1 for demo and 2 for entering your data ")
    if choice==1:
        trainClassifier()
        predictions=testClassify()
        with open(testf,'r') as testfile:
            outputResults(predictions,testfile)
        testfile.close()
        
    else:
        testvals=[]
        '''
        Taking user input
        '''
        print "** Please enter as shown in the examples because there are no input validations yet!:) **"
        resume='y' 
        while(resume=='y'):
            crop=raw_input("Enter the crop name. Your options are 'Cotton', 'None', 'Cumin', 'Mustard', 'Other', 'Wheat', 'Castor', 'Groundnut', 'Onion', 'Sorghum', 'Brinjal', 'Chilli', 'Gram', 'Paddy', 'Millet', 'Sesame', 'Banana', 'Maize', 'Garlic', 'Tobacco', 'Papaya'  ")
            dur=raw_input("Enter the average call duration in seconds Eg: 54,45,55  ")
            month=raw_input("Enter the month name in short. Eg: Jan, Feb, Mar  ")
            testvals.append([str(crop),int(dur),str(month)])
            resume=raw_input("Input more? y/n: ")
            resume=str(resume).lower()

        with open('topic_classifier.pkl', 'rb') as fid:
            topicClf = cPickle.load(fid)
        
        testdat=prepare(testvals)[0]
        predictions=topicClf.predict_proba(testdat)
        outputResults(predictions,testvals)
