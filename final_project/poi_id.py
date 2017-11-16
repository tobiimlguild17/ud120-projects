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
from sklearn.feature_selection import chi2
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

### Remove outliers
#data_no_outliers, labels_no_outliers = \
#    outliers.remove_outliers_based_on_std_deviations(data_all_features_scaled, labels, number_of_standard_deviations=2)

#data_no_outliers = np.abs(data_all_features) # <- this one has the best performance
data_no_outliers = data_all_features_scaled
labels_no_outliers = labels

feature_chi_scores = SelectKBest(chi2, k="all").fit(data_no_outliers, labels_no_outliers).scores_
print 'best features: '
print np.array(features_list)[np.argsort(feature_chi_scores) + 1]
print 'indices'
print np.argsort(feature_chi_scores)
print 'chi scores'
print np.sort(feature_chi_scores)

features_list = np.array(features_list)[np.argsort(feature_chi_scores) + 1][10:]
features_list = np.insert(features_list, 0, 'poi')
print "features_list", features_list

#features_formatted = np.transpose(np.squeeze(data_no_outliers))

# Plots
#Draw(features_formatted[9], features_formatted[4], labels_no_outliers, f1_name=features_list[10], f2_name=features_list[5])
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#from sklearn.pipeline import Pipeline
#from sklearn.decomposition import PCA
#estimators = [('reduce_dim', PCA(n_components=5)), ('clf', GaussianNB())]
#clf = Pipeline(estimators)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {"base_estimator__splitter" :   ["best", "random"],
              "base_estimator__max_depth" : [None, 3],
              "base_estimator__min_samples_split" : [2,5,10],
              "n_estimators": [32, 64],
              "learning_rate" : [0.1,1,10]}

dtc = DecisionTreeClassifier(criterion='entropy',max_features="auto")
adaclf = AdaBoostClassifier(base_estimator = dtc)
clf = GridSearchCV(adaclf, param_grid=param_grid, scoring = 'f1')

"""
Accuracy: 0.78560 Precision: 0.24106 Recall: 0.28300 F1: 0.26035 F2: 0.27348
Total predictions: 15000 True positives:  566 False positives: 1782 False negatives: 1434 True negatives: 11218
"""
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)