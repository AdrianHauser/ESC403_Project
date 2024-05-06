from pathlib import Path

import torch
from joblib import load
from src.feature_engineering.tasks.fire_direction import add_fire_direction_to_tensor
from src.feature_engineering.tasks.fire_distance_matrix import add_fire_distance_to_tensor
from src.feature_engineering.tasks.wind_aware_influence import add_flow_accumulation_to_tensor

from config import ROOT_DIR, DATA_TRIAL_DIR, DATA_PROCESSED_DIR, DATA_FEATURE_ENGINEERED_DIR
from src.data_processing.constants import INPUT_FEATURES
from src.feature_engineering import Pipeline
from utils import save_object


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

        # Save to Pickle
        save_path = DATA_TRIAL_DIR if self.testing else DATA_FEATURE_ENGINEERED_DIR
        save_object(X, path=save_path / Path(r"X_fe.pkl"))
        save_object(y, path=save_path / Path(r"y_fe.pkl"))


    def run(self, *args, **kwargs):
        setup_result = self._setup()
        run_result = self._run(**setup_result)


if __name__ == "__main__":
    pipeline = FeatureEngineering(testing=True)
    pipeline.run()
