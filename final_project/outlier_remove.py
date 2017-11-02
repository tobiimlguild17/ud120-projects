from sklearn.neighbors import LocalOutlierFactor
from helpers import Draw_outliers

### The LOF algorithm is an unsupervised outlier detection method which computes the local density deviation of a given data
### point with respect to its neighbors. It considers as outlier samples that have a substantially lower density than their neighbors.

def remove(features, features_formatted, neighbor):

    clf_outlier = LocalOutlierFactor(n_neighbors=neighbor,algorithm='auto')
    outliers = clf_outlier.fit_predict(features)
    Draw_outliers(features_formatted[0], features_formatted[1], outliers)
    features = [feature for [feature,outlier] in zip(features,outliers) if outlier == 1]
    return features