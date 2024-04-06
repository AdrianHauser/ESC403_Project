import numpy as np
import torch
import tensorflow as tf
import pickle
from tfrecord_reader import get_dataset
from scipy.ndimage import distance_transform_edt

def tf_to_torch(tf_tensor: tf.Tensor) -> torch.Tensor:
    """
    Utility function to convert tf.Tensor to pytorch.Tensor
    """
    np_array = tf_tensor.numpy()  # Convert TensorFlow tensor to NumPy array
    torch_tensor = torch.tensor(np_array)  # Convert NumPy array to PyTorch tensor

    return torch_tensor


def read_tfrecords(file_pattern: str, batch_size: int):
      """
      Takes file pattern and batch size and return X and y in torch.Tensor format
      """

      dataset = get_dataset(
            file_pattern,
            data_size=64,
            sample_size=64,
            batch_size=batch_size,
            num_in_channels=12,
            compression_type=None,
            clip_and_normalize=False,
            clip_and_rescale=False,
            random_crop=False,
            center_crop=False)

      # Get the inputs and labels separately, transform them to torch.Tensor
      X, y = next(iter(dataset.take(1)))
      X = tf_to_torch(X)
      y = tf_to_torch(y)
      return X, y


# Utility Function to save large objects in .pkl format
def save_object(obj, filename: str):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


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

def add_fire_distance_to_tensor(tensor: torch.Tensor, fire_mask_index: int) -> torch.Tensor:
    """
    Adds distance Matrix to a tensor of the shape (m,n,n,k)
    based on d-1 fire mask which is given with the mask index.
    """
    assert fire_mask_index < tensor.shape[3] # Checks the mask index is in bounds of the tensor
    distance_matrix = _compute_distance_matrix(tensor[:,:,:,fire_mask_index]) # Compute distance matrix on mask
    distance_tensor = torch.from_numpy(distance_matrix)
    distance_tensor = distance_tensor.unsqueeze(-1) # Unsqueeze adds dimension m to make it compatible
    tensor_with_distance = torch.cat([distance_tensor, tensor], dim=-1)

    return tensor_with_distance


def add_flow_accumulation_to_tensor(tensor: torch.Tensor, wind_index, fire_mask_index) -> torch.Tensor:
    return None