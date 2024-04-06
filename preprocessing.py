import numpy as np
import torch
import tensorflow as tf

def tf_to_torch(tf_tensor: tf.Tensor) -> torch.Tensor:
    """ Utility function to convert tf.Tensor to pytorch.Tensor """
    np_array = tf_tensor.numpy()  # Convert TensorFlow tensor to NumPy array
    torch_tensor = torch.tensor(np_array)  # Convert NumPy array to PyTorch tensor

    return torch_tensor
