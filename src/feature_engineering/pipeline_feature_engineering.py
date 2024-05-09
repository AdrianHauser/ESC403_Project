import os
import pickle
from pathlib import Path

import torch
from joblib import load

from config import DATA_FEATURE_ENGINEERED_DIR, DATA_PROCESSED_DIR, DATA_TRIAL_DIR
from src.data_preprocessing.tasks.constants import INPUT_FEATURES
from src.feature_engineering import Pipeline
from src.feature_engineering.tasks.fire_direction import add_fire_direction_to_tensor
from src.feature_engineering.tasks.fire_distance_matrix import (
    add_fire_distance_to_tensor,
)
from src.feature_engineering.tasks.wind_aware_influence import (
    add_flow_accumulation_to_tensor,
)


class FeatureEngineering(Pipeline):

    def __init__(self, testing):
        self.testing = testing

    def _setup(self) -> dict:
        """Setup phase."""
        read_dir = DATA_TRIAL_DIR if self.testing else DATA_PROCESSED_DIR

        return {
            "X": load(read_dir / Path(r"X_torch.pkl")),
            "y": load(read_dir / Path(r"y_torch.pkl")),
        }

    def _run(self, *args, **kwargs) -> torch.Tensor:
        """Run phase."""

        X = kwargs["X"]
        y = kwargs["y"]
        wind_direction_index = INPUT_FEATURES.index("th")
        wind_speed_index = INPUT_FEATURES.index("vs")
        fire_mask_index = INPUT_FEATURES.index("PrevFireMask")

        # Add Fire Distance Mask
        X = add_fire_distance_to_tensor(X, fire_mask_index)

        # Add flow accumulation
        X = add_flow_accumulation_to_tensor(
            X, wind_direction_index, wind_speed_index, fire_mask_index
        )

        # Add Fire Direction
        X = add_fire_direction_to_tensor(X, wind_direction_index, fire_mask_index)

        # Ensure the save directory exists, define the file paths
        save_dir = DATA_TRIAL_DIR if self.testing else DATA_FEATURE_ENGINEERED_DIR
        os.makedirs(save_dir, exist_ok=True)
        X_fe_path = save_dir / Path("X_fe.pkl")
        y_fe_path = save_dir / Path("y_fe.pkl")

        # Write X_torch, y_torch to file
        with open(X_fe_path, "wb") as f:
            pickle.dump(X, f)
        with open(y_fe_path, "wb") as f:
            pickle.dump(y, f)

    def run(self, *args, **kwargs):
        setup_result = self._setup()
        self._run(**setup_result)


if __name__ == "__main__":
    pipeline = FeatureEngineering(testing=True)
    pipeline.run()
