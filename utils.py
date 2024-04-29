import pickle

import tensorflow as tf
import torch


def save_object(obj, path: str):
    """Utility function to write an arbitrary python object to a Pickle file"""
    with open(path, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Written {type(obj)} to {path}.")


def tf_to_torch(tf_tensor: tf.Tensor) -> torch.Tensor:
    """Utility function to convert tf.Tensor to pytorch.Tensor"""
    np_array = tf_tensor.numpy()
    torch_tensor = torch.tensor(np_array)
    return torch_tensor
