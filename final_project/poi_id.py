#!/usr/bin/python

import sys
import numpy as np
import outliers
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from helpers import Draw

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

features_list = data_dict["TOTAL"].keys()
features_list.insert(0, features_list.pop(features_list.index("poi")))
features_list.pop(features_list.index("email_address"))
features_list.pop(features_list.index("loan_advances"))

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data_all_features_and_labels = featureFormat(my_dataset, features_list, sort_keys=True)
labels, data_all_features = targetFeatureSplit(data_all_features_and_labels)
labels = np.array(labels)

### Scale all the features
scaler = MinMaxScaler()
data_all_features_scaled = scaler.fit_transform(data_all_features)


### Find best features
feature_scores = SelectKBest(f_regression, k="all").fit(data_all_features_scaled, labels).scores_
number_of_features_to_keep = 8
features_list = np.array(features_list)[np.argsort(feature_scores) + 1][-number_of_features_to_keep:]
features_list = np.insert(features_list, 0, 'poi')

# Plots
#Draw(features_formatted[9], features_formatted[4], labels_no_outliers, f1_name=features_list[10], f2_name=features_list[5])
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
steps = [('transform', PCA(n_components=len(features_list)-1)), ('scaling', MinMaxScaler()), ('clf', GaussianNB())]
clf = Pipeline(steps)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features_list, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)