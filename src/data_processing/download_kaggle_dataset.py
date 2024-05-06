import json
import opendatasets as od
from config import DATA_DIR
from pathlib import Path
import os


def download_next_day_wildfire_spread():
    # Write kaggle.json file to circumvent manual login
    with open('kaggle.json', 'w') as file:
        json.dump({'username': '', 'key': ''}, file)

    dataset_url = "https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread"
    od.download(dataset_id_or_url = dataset_url,
                data_dir = DATA_DIR, # od.download creates a parent folder 'next-day-wildfire-spread'
                force = False)

    os.rename(
        DATA_DIR / Path(r'next-day-wildfire-spread'),
        DATA_DIR / Path(r'next_day_wildfire_spread')
    )

if __name__ == "__main__":
    download_next_day_wildfire_spread()

