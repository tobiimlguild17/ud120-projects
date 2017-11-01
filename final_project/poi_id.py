#!/usr/bin/python

import sys
import numpy as np
import outliers
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from helpers import Draw

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

features_list = data_dict["TOTAL"].keys()
features_list.insert(0, features_list.pop(features_list.index("poi")))
features_list.pop(features_list.index("email_address"))
print features_list


### Task 2: Remove outliers
data_dict.pop("TOTAL")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

#for key in my_dataset:
#    for key2 in my_dataset[key]:
#        print key2

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
data = np.abs(data)
data_no_outliers = outliers.remove_outliers_based_on_std_deviations(data, number_of_standard_deviations=2)
labels, features = targetFeatureSplit(data_no_outliers)

feature_chi_scores = SelectKBest(chi2, k="all").fit(data_no_outliers[:, 1:], data_no_outliers[:, 0]).scores_
print 'best features: '
print np.array(features_list)[np.argsort(feature_chi_scores) + 1]
print np.argsort(feature_chi_scores)

features_formatted = np.transpose(np.squeeze(features))

# Plots
Draw(features_formatted[12], features_formatted[18], labels, f1_name=features_list[13], f2_name=features_list[19])
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)