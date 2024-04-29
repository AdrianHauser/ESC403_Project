"""
Script to read TensorFlow records from the raw next_day_wildfire_spread folder, convert
to pytorch tensors and write to their respective pickle files
"""

from pathlib import Path

from tfrecord_reader import get_dataset

from config import ROOT_DIR
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


def _write_tfrecords_to_torch_pickle(X, y, testing = False):
    """
    Takes file pattern and batch size and return X and y in torch.Tensor format
    """
    X_torch, y_torch = tf_to_torch(X), tf_to_torch(y)

    save_path = r"Data/trial/" if testing else r"Data/processed/"
    save_object(X_torch, path=ROOT_DIR / Path(save_path, "X_torch.pkl"))
    save_object(y_torch, path=ROOT_DIR / Path(save_path, "y_torch.pkl"))


def parse_data_to_pickle(file_pattern: Path, batch_size: int, testing: bool = False):
    X, y = _read_tf_records(file_pattern=file_pattern, batch_size = batch_size)
    _write_tfrecords_to_torch_pickle(X, y, testing = testing)


if __name__ == "__main__":

    tf_records_path = ROOT_DIR / r"Data/next_day_wildfire_spread/next_day_wildfire_spread*"

    # Full parse of all files
    parse_data_to_pickle(file_pattern = tf_records_path,
                         batch_size = 25_000,
                         testing = False)

    # Small subset of data for trial purposes
    parse_data_to_pickle(file_pattern = tf_records_path,
                         batch_size = 16,
                         testing = True)
