"""
Contains all feature engineering functions.
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def _compute_distance_matrix(matrix: np.ndarray) -> np.ndarray:
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


def add_fire_distance_to_tensor(
    tensor: torch.Tensor, fire_mask_index: int
) -> torch.Tensor:
    """
    Adds distance Matrix to a tensor of the shape (m,n,n,k)
    based on d-1 fire mask which is given with the mask index.
    """
    assert (
        fire_mask_index < tensor.shape[3]
    )  # Checks the mask index is in bounds of the tensor

    tensor_shape = (64, 64, 1)
    distance_final = torch.empty((tensor.shape[0],) + tensor_shape)

    for i in range(tensor.shape[0]):

        fire_mask_matrix = tensor[i, :, :, fire_mask_index].numpy()
        distance_matrix = _compute_distance_matrix(
            fire_mask_matrix
        )  # Compute distance matrix on mask
        distance_tensor = torch.from_numpy(distance_matrix)
        distance_tensor = distance_tensor.unsqueeze(
            -1
        )  # Unsqueeze adds dimension m to make it compatible

        distance_final[i] = distance_tensor

    tensor = torch.cat([tensor, distance_final], dim=-1)
    return tensor


def add_flow_accumulation_to_tensor(
    tensor: torch.Tensor, wind_index, fire_mask_index
) -> torch.Tensor:
    return None
