import numpy as np
import torch

from src.data_processing.constants import DATA_STATS


def discretize_wind_direction(wind_direction):
    """
    Discretizes wind direction into 8 principal directions.
    Args:
    - wind_direction (np.ndarray): 2D array with wind directions in degrees (0-360).

    Returns:
    - np.ndarray: 2D array with wind directions discretized (0-7).
    """
    return np.floor(((wind_direction + 22.5) % 360) / 45).astype(int)


def normalize_wind_speed(wind_speed, speed_min=0, speed_max=1):
    """
    Normalizes wind speed within a range from 0 to 1.
    Args:
    - wind_speed (np.ndarray): 2D array with wind speeds.
    - speed_min (float): Observed minimum speed (for normalization).
    - speed_max (float): Observed maximum speed (for normalization).

    Returns:
    - np.ndarray: 2D array with normalized wind speeds.
    """
    return (wind_speed - speed_min) / (speed_max - speed_min)


def direction_to_delta(direction):
    """
    Maps discrete wind direction to changes in coordinates (row, col).
    """
    mapping = {
        0: (-1, 0),
        1: (-1, 1),
        2: (0, 1),
        3: (1, 1),
        4: (1, 0),
        5: (1, -1),
        6: (0, -1),
        7: (-1, -1),
    }
    return mapping.get(direction, (0, 0))


def spread_wind_influence(
    row, col, fire_mask, wind_direction, wind_speed, influence_matrix
):
    direction = wind_direction[row, col]
    speed = wind_speed[row, col]
    d_row, d_col = direction_to_delta(direction)
    new_row, new_col = row + d_row, col + d_col

    while 0 <= new_row < fire_mask.shape[0] and 0 <= new_col < fire_mask.shape[1]:
        if influence_matrix[row, col] != 0:
            current_value = min(influence_matrix[row, col] * speed, 2)
        else:
            current_value = speed

        if fire_mask[new_row, new_col] == 1:
            added_value = min(current_value + 1, 2)
        else:
            added_value = current_value

        if influence_matrix[new_row, new_col] < added_value:
            influence_matrix[new_row, new_col] = added_value

        row, col = new_row, new_col
        new_row, new_col = row + d_row, col + d_col


def compute_wind_influence(fire_mask, wind_direction, wind_speed):
    """
    Calculates wind influence starting from each point of fire.
    Each point of fire in the fire_mask generates its own matrix of influence.
    Final modification: sets fire points to the maximum value found in the final influence matrix.
    """
    rows, cols = fire_mask.shape
    final_influence = np.zeros_like(fire_mask, dtype=float)

    for row in range(rows):
        for col in range(cols):
            if fire_mask[row, col] == 1:
                influence_matrix = np.zeros_like(fire_mask, dtype=float)
                spread_wind_influence(
                    row, col, fire_mask, wind_direction, wind_speed, influence_matrix
                )
                final_influence += influence_matrix

    # Finds the maximum value in the final influence matrix
    # max_value = np.max(final_influence)

    # Sets all fire points to the maximum value found
    # final_influence[fire_mask == 1] = max_value

    return final_influence


def add_flow_accumulation_to_tensor(
    tensor: torch.Tensor,
    wind_direction_index: int,
    wind_speed_index: int,
    fire_mask_index: int,
) -> torch.Tensor:

    # Get Input Tensors
    discretized_wind_direction = discretize_wind_direction(
        tensor[:, :, :, wind_direction_index].numpy()
    )

    speed_min, speed_max = DATA_STATS.get("vs")[
        :2
    ]  # Get global min max wind speeds for normalization
    normalized_wind_speed = normalize_wind_speed(
        tensor[:, :, :, wind_speed_index].numpy(),
        speed_min=speed_min,
        speed_max=speed_max,
    )

    fire_mask = tensor[:, :, :, fire_mask_index].numpy()

    tensor_shape = (64, 64, 1)
    wind_influence_final = torch.empty((16,) + tensor_shape)

    for i in range(tensor.shape[0]):
        wind_influence = compute_wind_influence(
            fire_mask[i, :, :],
            discretized_wind_direction[i, :, :],
            normalized_wind_speed[i, :, :],
        )
        wind_influence = torch.from_numpy(wind_influence)
        wind_influence = wind_influence.unsqueeze(
            -1
        )  # Unsqueeze adds dimension m to make it compatible
        wind_influence_final[i] = wind_influence

    tensor = torch.cat([tensor, wind_influence_final], dim=-1)

    return tensor
