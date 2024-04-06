import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

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
                spread_wind_influence(row, col, fire_mask, wind_direction, wind_speed, influence_matrix)
                final_influence += influence_matrix
    
    # Finds the maximum value in the final influence matrix
    max_value = np.max(final_influence)
    
    # Sets all fire points to the maximum value found
    final_influence[fire_mask == 1] = max_value
    
    return final_influence

def spread_wind_influence(row, col, fire_mask, wind_direction, wind_speed, influence_matrix):
    direction = wind_direction[row, col]
    speed = wind_speed[row, col]
    d_row, d_col = direction_to_delta(direction)
    new_row, new_col = row + d_row, col + d_col

    if 0 <= new_row < fire_mask.shape[0] and 0 <= new_col < fire_mask.shape[1]:
        if fire_mask[new_row, new_col] == 0:  # Continues only if the new pixel is not already a point of fire
            if influence_matrix[new_row, new_col] == 0:  # Avoids infinite cycles by checking if the pixel has already been visited
                influence_matrix[new_row, new_col] = influence_matrix[row, col] * speed if influence_matrix[row, col] != 0 else speed
                spread_wind_influence(new_row, new_col, fire_mask, wind_direction, wind_speed, influence_matrix)

def direction_to_delta(direction):
    """
    Maps discrete wind direction to changes in coordinates (row, col).
    """
    mapping = {0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1),
               4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1)}
    return mapping.get(direction, (0, 0))


# We could also apply a smoothing with mooving window

