"""
Script to read TensorFlow records from the raw next_day_wildfire_spread folder, convert
to pytorch tensors and write to their respective pickle files
"""

import os
import pickle
from pathlib import Path

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, DATA_TRIAL_DIR
from src.data_preprocessing.tasks.tfrecord_reader import get_dataset
from utils import save_object, tf_to_torch


def _read_tf_records(file_pattern: str, batch_size: int):
    """Reads TensorFlow record files at returns X (explanatory) and y (independent) data."""

    dataset = get_dataset(
        file_pattern,
        data_size=64,
        sample_size=64,
        batch_size=batch_size,
        num_in_channels=12,
        clip_and_normalize=False,
        clip_and_rescale=False,
        random_crop=False,
    )

    X, y = next(iter(dataset.take(1)))
    return X, y


def _write_tfrecords_to_torch_pickle(X, y, testing=False):
    """
    Takes file pattern and batch size and return X and y in torch.Tensor format
    """
    X_torch, y_torch = tf_to_torch(X), tf_to_torch(y)

    # Ensure the save directory exists, define the file paths
    save_dir = DATA_TRIAL_DIR if testing else DATA_PROCESSED_DIR
    os.makedirs(save_dir, exist_ok=True)
    X_torch_path = save_dir / Path("X_torch.pkl")
    y_torch_path = save_dir / Path("y_torch.pkl")

    # Write X_torch, y_torch to file
    with open(X_torch_path, "wb") as f:
        pickle.dump(X_torch, f)
    with open(y_torch_path, "wb") as f:
        pickle.dump(y_torch, f)


def parse_data_to_pickle(file_pattern: Path, batch_size: int, testing: bool = False):
    X, y = _read_tf_records(file_pattern=file_pattern, batch_size=batch_size)
    _write_tfrecords_to_torch_pickle(X, y, testing=testing)


if __name__ == "__main__":

    tf_records_path = DATA_RAW_DIR / r"next_day_wildfire_spread*"

    # Full parse of all files
    parse_data_to_pickle(file_pattern=tf_records_path, batch_size=25_000, testing=False)

    # Small subset of data for trial purposes
    parse_data_to_pickle(file_pattern=tf_records_path, batch_size=16, testing=True)
