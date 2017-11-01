import numpy as np


def remove_outliers_based_on_std_deviations(data, number_of_standard_deviations=2):
    features = data[:, 1:3]
    std_deviations = np.std(features)
    std_deviation_scalar = np.sqrt(np.sum(std_deviations**2))
    features_zero_mean = features - np.mean(features)
    distances_from_mean = np.sqrt(np.sum(features_zero_mean**2, axis=1))
    return data[(distances_from_mean < std_deviation_scalar * number_of_standard_deviations)]