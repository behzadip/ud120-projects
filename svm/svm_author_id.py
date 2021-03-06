#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
mdl = SVC(kernel='linear', C=10000)

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100]

t0 = time()
mdl.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred = mdl.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"
print "accuracy", accuracy_score(pred, labels_test)

print pred[10]
print pred[26]
print pred[50]

print  np.count_nonzero(pred)
#########################################################

#training time: 127.915 s
#prediction time: 13.752 s
#accuracy 0.984072810011

