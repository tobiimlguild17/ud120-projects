import numpy as np


def remove_outliers_based_on_std_deviations(data, labels, number_of_standard_deviations=2):
    std_deviations = np.std(data, axis=0)
    std_deviation_scalar = np.sqrt(np.sum(std_deviations**2))
    data_zero_mean = data - np.mean(data, axis=0)
    distances_from_mean = np.sqrt(np.sum(data_zero_mean**2, axis=1))
    indices_to_keep = (distances_from_mean < std_deviation_scalar * number_of_standard_deviations)
    return data[indices_to_keep], labels[indices_to_keep]