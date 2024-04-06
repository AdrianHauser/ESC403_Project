import numpy as np
import torch
import tensorflow as tf
import pickle
from tfrecord_reader import get_dataset

def tf_to_torch(tf_tensor: tf.Tensor) -> torch.Tensor:
    """ Utility function to convert tf.Tensor to pytorch.Tensor """
    np_array = tf_tensor.numpy()  # Convert TensorFlow tensor to NumPy array
    torch_tensor = torch.tensor(np_array)  # Convert NumPy array to PyTorch tensor

    return torch_tensor


def read_tfrecords(
        file_pattern: str = "Data/next_day_wildfire_spread/next_day_wildfire_spread*",
        batch_size: int = 200
):
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

      # get the inputs and labels separately
      X, y = next(iter(dataset.take(1)))
      X = tf_to_torch(X)
      y = tf_to_torch(y)
      return X, y


# Utility Function to save large objects in .pkl format
def save_object(obj, filename: str):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)