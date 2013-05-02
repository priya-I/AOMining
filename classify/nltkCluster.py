'''
Created on May 1, 2013

@author: priya

'''
from nltk import cluster
from nltk import api
from nltk import util
import numpy
from nltk.cluster import KMeansClusterer, euclidean_distance

users=[]
myMatrix=[]
inputfile="cluster_sample.csv"
villagef="villageslist.csv"
i=0
gd=3651201
gs=131070
villages={}
with open(villagef,"r") as vf:
    for row in vf:
        vill=row.strip("\r\n")
        villages[vill]=i
        i+=1

#uid,durations,sessions,age,treatment,educ,village
with open(inputfile,"r") as inputf:
    for row in inputf:
       # print row
        record=row.split(",")
        userid=record[0].strip("\n")
        duration=record[1]
        session=record[2]
        age=record[3]
        trgrp=record[4]
        educ=record[5]
        village=record[6].strip("\r\n")
        avg=record[7]
        print village
        ps=(float(session)/gs)*100
        pd=(float(duration)/gd)*100
        vilcode=villages[village]
        users.append(userid)
        myMatrix.append([pd,ps,float(age),int(trgrp),int(educ),int(vilcode)])
        #myMatrix.append([float(avg),float(age),int(trgrp),int(educ),int(vilcode)])
     
print myMatrix   
inputf.close()


clusterer=cluster.KMeansClusterer(3,euclidean_distance)
vectors=[numpy.array(f) for f in myMatrix]
clusters=clusterer.cluster(vectors,assign_clusters=True,trace=False)



print 'Means: ', clusterer.means()
for cl in clusters:
    print cl

#print clusterer.dendrogram().show()


  

