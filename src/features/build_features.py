import torch
import tensorflow as tf
import pickle
import joblib
from config import ROOT_DIR
from pathlib import Path

X_torch = joblib.load(ROOT_DIR / Path("Data/interim/X_torch.pkl"))
y_torch = joblib.load(ROOT_DIR / Path("Data/interim/y_torch.pkl"))

fire_mask_index = 11
X = add_fire_distance_to_tensor(X, fire_mask_index)

wind_direction_index = 5
wind_speed_index = 8
X = add_flow_accumulation_to_tensor(X, wind_direction_index, wind_speed_index, fire_mask_index)





