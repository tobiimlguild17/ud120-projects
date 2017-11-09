import numpy as np


def remove_outliers_based_on_std_deviations(features, labels, number_of_standard_deviations=2):
    std_deviations = np.std(features, axis=0)
    std_deviation_scalar = np.sqrt(np.sum(std_deviations**2))
    features_zero_mean = features - np.mean(features, axis=0)
    distances_from_mean = np.sqrt(np.sum(features_zero_mean**2, axis=1))
    indices_to_keep = (distances_from_mean < std_deviation_scalar * number_of_standard_deviations)
    print np.array(labels)[(distances_from_mean >= std_deviation_scalar * number_of_standard_deviations)]
    return np.array(features)[indices_to_keep], np.array(labels)[indices_to_keep]
