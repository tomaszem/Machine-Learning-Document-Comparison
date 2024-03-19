from sklearn.neighbors import NearestNeighbors
import numpy as np


"""
Calculates a suggested range for the DBSCAN eps parameter based on the k-nearest neighbors distances.

This function analyzes the distances to the nearest neighbors in the given dataset to suggest
a range for the eps parameter of the DBSCAN algorithm. It uses percentiles of these distances
to determine a starting and ending value for eps, as well as providing a set of suggested values
within this range.

Parameters:
- data: Numpy array containing the dataset.
- start_percentile: The percentile to start the eps range suggestion.
- end_percentile: The percentile to end the eps range suggestion.
- step: The step size to increase the percentile for generating suggested eps values.
"""

def find_optimal_eps_range(data, start_percentile=10, end_percentile=90, step=5):
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    # Sorting distances
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    # Calculating the range based on percentiles
    start_eps = np.percentile(distances, start_percentile)
    end_eps = np.percentile(distances, end_percentile)

    # Adjusting the range based on the step size
    suggested_eps_values = []
    for percentile in range(start_percentile, end_percentile + step, step):
        suggested_eps_values.append(np.percentile(distances, percentile))

    return start_eps, end_eps, suggested_eps_values
