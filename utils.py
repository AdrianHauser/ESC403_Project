import os
import pickle
import shutil

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


def move_all_files_from_folder(source, destination):
    files = os.listdir(source)
    file_count = len(files)

    if len(list) == 0:
        print(f"No files in source {source}")
    else:
        for file in files:
            file_name = os.path.join(source, file)
            shutil.move(file_name, destination)
        print(f"{file_count} files moved to {destination}")
