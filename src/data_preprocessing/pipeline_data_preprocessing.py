from pathlib import Path

import torch

from config import DATA_RAW_DIR
from src.data_preprocessing import Pipeline
from src.data_preprocessing.tasks.download_kaggle_dataset import (
    download_next_day_wildfire_spread,
)
from src.data_preprocessing.tasks.parse_data_to_pickle import parse_data_to_pickle


class DataPreprocessing(Pipeline):

    def __init__(self, download: bool):
        self.download = download

    def _setup(self):
        """If required, download the Kaggle Dataset"""
        if self.download:
            download_next_day_wildfire_spread()

    def _run(self, *args, **kwargs) -> torch.Tensor:
        """Run phase."""

        tf_records_path = DATA_RAW_DIR / r"next_day_wildfire_spread*"

        # Full parse of all files
        parse_data_to_pickle(
            file_pattern=tf_records_path, batch_size=25_000, testing=False
        )

        # Small subset of data for trial purposes
        parse_data_to_pickle(file_pattern=tf_records_path, batch_size=16, testing=True)

    def run(self, *args, **kwargs):
        self._setup()
        self._run()


if __name__ == "__main__":
    pipeline = DataPreprocessing(download=False)
    pipeline.run()
