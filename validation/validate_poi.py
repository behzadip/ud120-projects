#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import numpy as np
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import tree
from time import time
from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
print "prediction time:", round(time()-t0, 3), "s"

pred = clf.predict(features_test)
print "accuracy", clf.score(features_test, labels_test)

#print sum(pred)
#print len(pred)
#print sum(labels_test)

print np.dot(labels_test, pred)
print metrics.precision_score(labels_test, pred)
print metrics.recall_score(labels_test, pred)