import numpy as np
from scipy.ndimage import distance_transform_edt

def transform_matrix(matrix):
    """
    Transforms a 64x64 matrix of 0s and 1s into a matrix where 0s are replaced
    with the Euclidean distance to the nearest 1.

    Args:
    - matrix (np.ndarray): A 64x64 matrix of 0s and 1s.

    Returns:
    - np.ndarray: The transformed matrix.
    """
    inverted_matrix = 1 - matrix
    distance_matrix = distance_transform_edt(inverted_matrix)
    distance_matrix[matrix == 1] = 0

    return distance_matrix