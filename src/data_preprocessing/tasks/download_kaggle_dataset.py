import json
import os
from pathlib import Path

import opendatasets as od

from config import DATA_RAW_DIR
from utils import move_all_files_from_folder

DATASET_URL = r"https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread"


def _all_files_exist(path: str) -> bool:
    """Tests if all next-day-wildfire-spread .tfrecord files were downloaded"""

    phases = ["eval", "test", "train"]
    numbers = {
        "eval": ["00", "01"],
        "test": ["00", "01"],
        "train": [f"{i:02d}" for i in range(15)],
    }

    files_exist = []
    for phase in phases:
        for number in numbers[phase]:
            filename = f"next_day_wildfire_spread_{phase}_{number}.tfrecord"
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                files_exist.append(filename)

    return all(files_exist)


def download_next_day_wildfire_spread():
    """Download next-day-wildfire spread dataset from Kaggle"""

    if not _all_files_exist(DATA_RAW_DIR):

        # Write kaggle.json file to circumvent manual login
        with open("kaggle.json", "w") as file:
            json.dump({"username": "", "key": ""}, file)

        # Download dataset from Kaggle, creates a sub-folder 'next-day-wildfire-spread'
        od.download(
            dataset_id_or_url=DATASET_URL, data_dir=DATA_RAW_DIR, force=False  #
        )

        # Move the contents of the sub-folder to the parent directory and delete the sub-folder
        sub_folder_path = DATA_RAW_DIR / Path(r"next-day-wildfire-spread")
        move_all_files_from_folder(source=sub_folder_path, destination=DATA_RAW_DIR)
        os.rmdir(sub_folder_path)

        assert _all_files_exist(DATA_RAW_DIR)

    else:
        print("Data already exists. Skipping download.")


if __name__ == "__main__":
    download_next_day_wildfire_spread()
